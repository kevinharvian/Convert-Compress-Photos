# app.py
import io, os, zipfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import fitz  # PyMuPDF

# ===== HEIC/HEIF =====
HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

# ===== PAGE =====
st.set_page_config(page_title="Multi-ZIP → JPG & Kompres 50–150 KB", page_icon="📦", layout="wide")
st.title("📦 Multi-ZIP / Files → JPG & Kompres 50–150 KB")
st.caption("Konversi gambar & PDF ke JPG, kompres 50–150 KB, kualitas tajam. Hasil: 1 master ZIP berisi 1 folder: <nama_asli>_compressed.")

# ===== Sidebar Settings =====
with st.sidebar:
    st.header("⚙️ Pengaturan")
    SPEED_PRESET = st.selectbox("Preset kecepatan", ["fast", "balanced"], index=0)
    TARGET_KB = st.slider("Target maksimum (KB)", 60, 400, 150, 5)
    MIN_KB = st.slider("Target minimum (KB)", 20, 200, 50, 5)
    MIN_SIDE_PX = st.number_input("Sisi terpendek minimum (px)", 64, 2048, 256, 32)
    SCALE_MIN = st.slider("Skala minimum saat downscale", 0.10, 0.75, 0.35, 0.05)
    UPSCALE_MAX = st.slider("Batas upscale maksimum", 1.0, 3.0, 2.0, 0.1)
    SHARPEN_ON_RESIZE = st.checkbox("Sharpen ringan setelah resize", True)
    SHARPEN_AMOUNT = st.slider("Sharpen amount", 0.0, 2.0, 1.0, 0.1)
    PDF_DPI = 150 if SPEED_PRESET == "fast" else 200
    MASTER_ZIP_NAME = st.text_input("Nama master ZIP (tanpa unik suffix)", "compressed.zip")

MAX_QUALITY = 95
MIN_QUALITY = 15
BG_FOR_ALPHA = (255, 255, 255)
THREADS = max(2, (os.cpu_count() or 2))
ZIP_COMP_ALGO = zipfile.ZIP_STORED if SPEED_PRESET == "fast" else zipfile.ZIP_DEFLATED

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
PDF_EXT = {".pdf"}

# ===== Helpers (quality tuned like CODE 1) =====
def maybe_sharpen(img: Image.Image, do_it=True, amount=1.0) -> Image.Image:
    if not do_it or amount <= 0: return img
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(150*amount), threshold=2))

def to_rgb_flat(img: Image.Image, bg=BG_FOR_ALPHA) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.convert("RGBA").split()[-1])
        return base
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def save_jpg_bytes(img: Image.Image, quality: int) -> bytes:
    # optimize/progressive = True (kualitas lebih baik pada ukuran sama)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)
    return buf.getvalue()

def try_quality_bs(img: Image.Image, target_kb: int, q_min=MIN_QUALITY, q_max=MAX_QUALITY):
    lo, hi = q_min, q_max
    best_bytes = None
    best_q = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = save_jpg_bytes(img, mid)
        if len(data) <= target_kb * 1024:
            best_bytes, best_q = data, mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_bytes, best_q

def resize_to_scale(img: Image.Image, scale: float, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    nw, nh = max(int(w*scale),1), max(int(h*scale),1)
    out = img.resize((nw, nh), Image.LANCZOS)
    return maybe_sharpen(out, do_sharpen, amount)

def ensure_min_side(img: Image.Image, min_side_px: int, do_sharpen=True, amount=1.0) -> Image.Image:
    w, h = img.size
    if min(w, h) >= min_side_px: return img
    scale = min_side_px / min(w, h)
    return resize_to_scale(img, scale, do_sharpen, amount)

def load_image_from_bytes(name: str, raw: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(im)

def gif_first_frame(im: Image.Image) -> Image.Image:
    try: im.seek(0)
    except Exception: pass
    return im.convert("RGBA") if im.mode == "P" else im

def compress_into_range(
    base_img: Image.Image,
    min_kb: int,
    max_kb: int,
    min_side_px: int,
    scale_min: float,
    upscale_max: float,
    do_sharpen: bool,
    sharpen_amount: float,
):
    base = to_rgb_flat(base_img)

    # A. tanpa resize
    data, q = try_quality_bs(base, max_kb)
    if data is not None:
        result = (data, 1.0, q, len(data))
    else:
        # B. downscale cepat (few iterations)
        lo, hi = scale_min, 1.0
        best_pack = None
        for _ in range(8 if SPEED_PRESET == "fast" else 12):
            mid = (lo + hi) / 2
            candidate = resize_to_scale(base, mid, do_sharpen, sharpen_amount)
            candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
            d, q2 = try_quality_bs(candidate, max_kb)
            if d is not None:
                best_pack = (d, mid, q2, len(d))
                lo = mid + (hi - mid) * 0.35
            else:
                hi = mid - (mid - lo) * 0.35
            if hi - lo < 1e-3:
                break
        if best_pack is None:
            smallest = resize_to_scale(base, scale_min, do_sharpen, sharpen_amount)
            smallest = ensure_min_side(smallest, min_side_px, do_sharpen, sharpen_amount)
            d = save_jpg_bytes(smallest, MIN_QUALITY)
            result = (d, scale_min, MIN_QUALITY, len(d))
        else:
            result = best_pack

    data, scale_used, q_used, size_b = result

    # C. kalau masih < min_kb → naikkan kualitas/detail (tanpa lewat target)
    if size_b < min_kb * 1024:
        img_now = resize_to_scale(base, scale_used, do_sharpen, sharpen_amount)
        img_now = ensure_min_side(img_now, min_side_px, do_sharpen, sharpen_amount)
        d, q2 = try_quality_bs(img_now, max_kb, max(q_used, MIN_QUALITY), MAX_QUALITY)
        if d is not None and len(d) > size_b:
            data, q_used, size_b = d, q2, len(d)

        cur_scale = scale_used
        iters = 0
        max_iters = 6 if SPEED_PRESET == "fast" else 12
        while size_b < min_kb * 1024 and cur_scale < upscale_max and iters < max_iters:
            cur_scale = min(cur_scale * 1.2, upscale_max)
            candidate = resize_to_scale(base, cur_scale, do_sharpen, sharpen_amount)
            candidate = ensure_min_side(candidate, min_side_px, do_sharpen, sharpen_amount)
            d, q3 = try_quality_bs(candidate, max_kb, MIN_QUALITY, MAX_QUALITY)
            if d is None:
                cur_scale *= 0.95
                iters += 1
                continue
            if len(d) > size_b:
                data, q_used, size_b, scale_used = d, q3, len(d), cur_scale
            iters += 1

    return data, scale_used, q_used, size_b

# ===== PDF → PIL via PyMuPDF =====
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(ImageOps.exif_transpose(img))
    return images

# ===== ZIP/Jobs Handling =====
def extract_zip_to_memory(zf_bytes: bytes) -> List[Tuple[Path, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zf_bytes), 'r') as zf:
        for info in zf.infolist():
            if info.is_dir(): continue
            with zf.open(info, 'r') as f:
                data = f.read()
            out.append((Path(info.filename), data))
    return out

def guess_base_name_from_zip(zipname: str) -> str:
    base = Path(zipname).stem
    return base or "output"

def process_one_file_entry(relpath: Path, raw_bytes: bytes, input_root_label: str):
    processed = []
    skipped = []
    outputs: Dict[str, bytes] = {}

    ext = relpath.suffix.lower()
    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                try:
                    data, scale, q, size_b = compress_into_range(
                        pil_img, MIN_KB, TARGET_KB, MIN_SIDE_PX,
                        SCALE_MIN, UPSCALE_MAX, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
                    )
                    out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                    outputs[out_rel] = data
                    processed.append((out_rel, size_b, scale, q, MIN_KB*1024 <= size_b <= TARGET_KB*1024))
                except Exception as e:
                    skipped.append((f"{relpath} (page {idx})", str(e)))
        elif ext in IMG_EXT and (ext not in {".heic", ".heif"} or HEIF_OK):
            im = load_image_from_bytes(relpath.name, raw_bytes)
            if ext == ".gif":
                im = gif_first_frame(im)
            data, scale, q, size_b = compress_into_range(
                im, MIN_KB, TARGET_KB, MIN_SIDE_PX,
                SCALE_MIN, UPSCALE_MAX, SHARPEN_ON_RESIZE, SHARPEN_AMOUNT
            )
            out_rel = relpath.with_suffix(".jpg").as_posix()
            outputs[out_rel] = data
            processed.append((out_rel, size_b, scale, q, MIN_KB*1024 <= size_b <= TARGET_KB*1024))
        elif ext in {".heic", ".heif"} and not HEIF_OK:
            skipped.append((str(relpath), "Butuh pillow-heif (tidak tersedia)"))
        # else: skip
    except Exception as e:
        skipped.append((str(relpath), str(e)))

    return input_root_label, processed, skipped, outputs

def add_to_master_zip(master: zipfile.ZipFile, top_folder: str, arcname: str, data: bytes):
    master.writestr(f"{top_folder}/{arcname}", data)

def unique_arcname(name: str, used: set) -> str:
    """Pastikan nama file unik di dalam ZIP (flat)."""
    base, ext = os.path.splitext(name)
    cand = name
    i = 2
    while cand in used:
        cand = f"{base}_{i}{ext}"
        i += 1
    used.add(cand)
    return cand

# ===== UI Upload & Run =====
st.subheader("1) Upload ZIP atau File Lepas")
uploaded_files = st.file_uploader(
    "Upload beberapa ZIP (berisi folder/gambar/PDF) dan/atau file lepas (gambar/PDF).",
    type=None, accept_multiple_files=True
)

run = st.button("🚀 Proses & Buat Master ZIP", type="primary", disabled=not uploaded_files)

if run:
    if not uploaded_files:
        st.warning("Silakan upload minimal satu file."); st.stop()

    # Build jobs: tiap ZIP = 1 job; file lepas digabung jadi 1 job
    jobs = []
    used_labels = set()

    def unique_name(base: str, used: set) -> str:
        name = base; idx = 2
        while name in used:
            name = f"{base}_{idx}"; idx += 1
        used.add(name); return name

    zip_inputs, loose_inputs = [], []
    for f in uploaded_files:
        name, raw = f.name, f.read()
        if name.lower().endswith(".zip"): zip_inputs.append((name, raw))
        else: loose_inputs.append((name, raw))

    for zname, zbytes in zip_inputs:
        try:
            pairs = extract_zip_to_memory(zbytes)  # [(relpath, data)]
            base_label = unique_name(guess_base_name_from_zip(zname), used_labels)
            items = [(relp, data) for (relp, data) in pairs if relp.suffix.lower() in IMG_EXT.union(PDF_EXT)]
            if items:
                jobs.append({"label": base_label, "items": items})
        except Exception as e:
            st.error(f"Gagal membuka ZIP {zname}: {e}")

    if loose_inputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_label = unique_name(f"compressed_pict_{ts}", used_labels)
        items = [(Path(name), data) for (name, data) in loose_inputs]
        jobs.append({"label": base_label, "items": items})

    if not jobs:
        st.error("Tidak ada berkas valid (butuh gambar/PDF, atau ZIP berisi file-file tersebut)."); st.stop()

    st.write(f"🔧 Ditemukan **{sum(len(j['items']) for j in jobs)}** berkas dari **{len(jobs)}** input.")

    # Proses multi-core
    summary: Dict[str, List[Tuple[str, int, float, int, bool]]] = defaultdict(list)
    skipped_all: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    per_job_outputs: Dict[str, Dict[str, bytes]] = defaultdict(dict)

    def worker(label: str, relp: Path, raw: bytes):
        return process_one_file_entry(relp, raw, label)

    all_tasks = [(job["label"], relp, data) for job in jobs for (relp, data) in job["items"]]
    total, done, processed_total = len(all_tasks), 0, 0
    progress = st.progress(0.0)

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [ex.submit(worker, *t) for t in all_tasks]
        for fut in as_completed(futures):
            label, prc, skp, outs = fut.result()
            summary[label].extend(prc)
            skipped_all[label].extend(skp)
            if outs:
                per_job_outputs[label].update(outs)
            processed_total += len(prc)
            done += 1
            progress.progress(min(done/total, 1.0))

    # ==== Nama ZIP unik (…_1.zip, …_2.zip, dst) ====
    if "zip_counter" not in st.session_state:
        st.session_state.zip_counter = 0
    st.session_state.zip_counter += 1

    base_zip_name = (MASTER_ZIP_NAME.strip() or "compressed.zip")
    stem = Path(base_zip_name).stem
    ext = ".zip"
    final_zip_name = f"{stem}_{st.session_state.zip_counter}{ext}"

    # ==== Folder di dalam ZIP: sesuai "nama asli folder" + _compressed
    # Ambil dari job pertama (sesuai permintaan).
    top_folder = f"{jobs[0]['label']}_compressed"

    # ==== Build master ZIP in-memory (FLAT, 1 folder saja) ====
    used_arc = set()
    master_buf = io.BytesIO()
    with zipfile.ZipFile(master_buf, "w", compression=ZIP_COMP_ALGO) as master:
        # pastikan folder top-level ada
        master.writestr(f"{top_folder}/", "")
        for job in jobs:
            base = job["label"]
            for rel_path, *_ in summary[base]:
                if rel_path in per_job_outputs[base]:
                    # flatten: ambil hanya nama file (tanpa subfolder)
                    name_only = Path(rel_path).name
                    # jika bukan job pertama, prefiks label biar jelas & tidak tabrakan
                    if base != jobs[0]["label"]:
                        name_only = f"{base}_{name_only}"
                    # pastikan unik di dalam ZIP
                    arcname = unique_arcname(name_only, used_arc)
                    add_to_master_zip(master, top_folder, arcname, per_job_outputs[base][rel_path])
    master_buf.seek(0)

    # Ringkasan
    st.subheader("📊 Ringkasan")
    grand_ok = 0; grand_cnt = 0
    for job in jobs:
        base = job["label"]; items = summary[base]; skipped = skipped_all[base]
        with st.expander(f"📦 {base} — {len(items)} file diproses, {len(skipped)} dilewati/errored"):
            ok = 0
            for name, size_b, scale, q, in_range in items:
                kb = size_b/1024
                flag = "✅" if in_range else ("🔼" if kb < MIN_KB else "⚠️")
                st.write(f"{flag} `{name}` → **{kb:.1f} KB** | scale≈{scale:.3f} | quality={q}")
                ok += 1 if in_range else 0
            if skipped:
                st.write("**Dilewati/Errored:**")
                for n, reason in skipped[:30]:
                    st.write(f"- {n}: {reason}")
            st.caption(f"Berhasil di rentang {MIN_KB}–{TARGET_KB} KB: **{ok}/{len(items)}**")
        grand_ok += ok; grand_cnt += len(items)

    st.write("---")
    st.write(f"**Total file OK di rentang:** {grand_ok}/{grand_cnt} | **Total berkas diproses:** {processed_total}")

    st.download_button(
        "⬇️ Download Master ZIP",
        data=master_buf.getvalue(),
        file_name=final_zip_name,  # unik: compressed_1.zip, compressed_2.zip, ...
        mime="application/zip",
    )
    st.success(f"Selesai! File siap diunduh: {final_zip_name} (folder di dalam: {top_folder}/)")
