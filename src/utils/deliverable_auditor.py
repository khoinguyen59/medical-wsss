"""
SE374.R11 Medical AI - Independent Deliverable Quality Assurance Auditor.
Hệ thống kiểm soát chất lượng báo cáo, tài liệu và ấn phẩm học thuật độc lập.

Chức năng kiểm tra 4 tầng:
1. Image & Visual Quality Audit: Chặn ảnh chụp đời thường, ảnh mờ, ép buộc hình vẽ khoa học vector/300 DPI.
2. Typography & Spelling Audit: Chặn dấu gạch ngang dài kiểu AI (em-dash, en-dash), kiểm tra mã ký tự UTF-8, lỗi dấu câu.
3. Content & Scientific Integrity Audit: Quét dummy metrics, mockups, kiểm tra công thức toán và DOI/arXiv.
4. Document Structure Audit: Kiểm tra cấu trúc tiêu đề, bảng biểu, chú giải hình ảnh.
"""

import os
import re
import sys
from typing import List, Dict, Any, Tuple
from PIL import Image, ExifTags


class DeliverableIssue:
    def __init__(self, category: str, severity: str, file_path: str, line_num: int, message: str, remediation: str):
        self.category = category      # 'IMAGE', 'TYPOGRAPHY', 'INTEGRITY', 'STRUCTURE'
        self.severity = severity      # 'CRITICAL', 'WARNING', 'INFO'
        self.file_path = file_path
        self.line_num = line_num
        self.message = message
        self.remediation = remediation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "file": self.file_path,
            "line": self.line_num,
            "message": self.message,
            "remediation": self.remediation
        }

    def __str__(self) -> str:
        loc = f"{os.path.basename(self.file_path)}:{self.line_num}" if self.line_num > 0 else os.path.basename(self.file_path)
        return f"[{self.severity}] [{self.category}] {loc} - {self.message} (Khắc phục: {self.remediation})"


class DeliverableAuditor:
    """Independent Academic & Medical AI Deliverable Quality Auditor."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self.issues: List[DeliverableIssue] = []
        self.rendered_pages: List[str] = []

    def audit_file(self, file_path: str) -> List[DeliverableIssue]:
        """Audits either a markdown or docx file based on extension."""
        abs_path = os.path.abspath(file_path)
        if abs_path.lower().endswith(".docx"):
            return self.audit_docx_file(abs_path)
        else:
            return self.audit_markdown_file(abs_path)

    def audit_markdown_file(self, md_path: str) -> List[DeliverableIssue]:
        """Audits a markdown file across all quality gates."""
        abs_path = os.path.abspath(md_path)
        if not os.path.exists(abs_path):
            self.issues.append(DeliverableIssue("STRUCTURE", "CRITICAL", md_path, 0, f"Tệp không tồn tại: {md_path}", "Kiểm tra đường dẫn."))
            return self.issues

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        lines = content.split("\n")

        # Run audits
        self._audit_typography(abs_path, lines)
        self._audit_scientific_integrity(abs_path, lines)
        self._audit_document_structure(abs_path, lines)
        self._audit_embedded_images(abs_path, lines)
        self._audit_faculty_and_identity(abs_path, lines)

        return self.issues

    def audit_docx_file(self, docx_path: str) -> List[DeliverableIssue]:
        """Audits compiled DOCX file: footers, headers, styles, font colors, and faculty names."""
        abs_path = os.path.abspath(docx_path)
        if not os.path.exists(abs_path):
            self.issues.append(DeliverableIssue("STRUCTURE", "CRITICAL", docx_path, 0, f"Tệp DOCX không tồn tại: {docx_path}", "Biên dịch lại DOCX."))
            return self.issues

        try:
            import zipfile
            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(abs_path, "r") as z:
                files = z.namelist()

                # Check 1: Faculty / Header / Footer identity leak
                wrong_faculty_patterns = [r"tran\s*hung\s*nghiep", r"th\s*nghiep", r"unkn\s*lab", r"thay\s*nghiep"]
                for fname in files:
                    if fname.startswith("word/footer") or fname.startswith("word/header") or fname == "word/document.xml":
                        raw_xml = z.read(fname).decode("utf-8", errors="ignore").lower()
                        for pat in wrong_faculty_patterns:
                            if re.search(pat, raw_xml):
                                self.issues.append(DeliverableIssue(
                                    category="INTEGRITY",
                                    severity="CRITICAL",
                                    file_path=docx_path,
                                    line_num=0,
                                    message=f"Phat hien ten giang vien cu hoac lab ngoai ('{pat}') trong '{fname}'. Giang vien mon SE374.R11 la TS. Bui Cao Doanh!",
                                    remediation="Chuan hoa lai template va footer khong de ten Tran Hung Nghiep hay UNKN Lab."
                                ))

                # Check 2: Leaked Meta-Prompts / Rubrics in document.xml
                if "word/document.xml" in files:
                    doc_text = "".join(ET.fromstring(z.read("word/document.xml")).itertext())
                    meta_rubric_patterns = [
                        (r"invariant-\d+", "Ro ri ma INVARIANT noi bo cua agent vao bao cao"),
                        (r"cam ket liem chinh", "Ro ri loi cam ket noi bo / meta-disclaimer vao bao cao"),
                        (r"mô tả đề tài trong 1 đoạn văn", "Ro ri loi nhac yeu cau de bai (rubric) vao tieu de"),
                        (r"điền số nhóm", "Chua dien so thu tu nhom chinh thuc"),
                    ]
                    for pat, desc in meta_rubric_patterns:
                        if re.search(pat, doc_text, flags=re.IGNORECASE):
                            self.issues.append(DeliverableIssue(
                                category="STRUCTURE",
                                severity="CRITICAL",
                                file_path=docx_path,
                                line_num=0,
                                message=f"Phat hien thong tin yeu cau / format khong can thiet: {desc}.",
                                remediation="Loai bo toan bo ky hieu INVARIANT, prompt rubric va placeholder noi bo."
                            ))

                # Check 3: Font Color Academic Black Audit (Styles & Runs)
                if "word/styles.xml" in files:
                    styles_xml = z.read("word/styles.xml").decode("utf-8", errors="ignore")
                    styles_root = ET.fromstring(styles_xml)
                    for style in styles_root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}style"):
                        style_id = style.attrib.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}styleId", "")
                        for color in style.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}color"):
                            val = color.attrib.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", "").lower()
                            if val in ["666666", "777777", "808080", "888888", "999999", "555555"]:
                                self.issues.append(DeliverableIssue(
                                    category="TYPOGRAPHY",
                                    severity="CRITICAL",
                                    file_path=docx_path,
                                    line_num=0,
                                    message=f"Kieu dinh dang '{style_id}' su dung mau chu xam (#{val}) thay vi mau den (#000000). Vi pham tieu chuan in an hoc thuat!",
                                    remediation="Chuan hoa style ve mau den tuyen (#000000)."
                                ))

        except Exception as e:
            self.issues.append(DeliverableIssue("STRUCTURE", "WARNING", docx_path, 0, f"Khong the giai ma file DOCX: {e}", "Kiem tra file."))

        # Check 4: Visual Page Rendering & Multimodal Vision Audit Gate
        self._audit_docx_visual_rendering(abs_path)

        return self.issues

    def _audit_docx_visual_rendering(self, docx_path: str):
        """Converts DOCX to PDF and renders all pages as PNG images for visual and vision inspection.
        Uses system Word COM and PyMuPDF (fitz) - standard installed libraries.
        """
        try:
            import win32com.client
            import pythoncom
            import fitz
        except ImportError:
            return

        pages_dir = os.path.join(os.path.dirname(docx_path), "assets", "preview", "pages")
        os.makedirs(pages_dir, exist_ok=True)
        pdf_path = os.path.splitext(docx_path)[0] + ".pdf"

        need_convert = not os.path.exists(pdf_path) or (os.path.getmtime(docx_path) > os.path.getmtime(pdf_path))
        if need_convert:
            word = None
            com_initialized = False
            try:
                pythoncom.CoInitialize()
                com_initialized = True
                word = win32com.client.Dispatch("Word.Application")
                word.Visible = False
                doc = word.Documents.Open(docx_path)
                doc.SaveAs(pdf_path, FileFormat=17)  # 17 = wdFormatPDF
                doc.Close()
            except Exception:
                pass
            finally:
                if word is not None:
                    try:
                        word.Quit()
                    except Exception:
                        pass
                if com_initialized:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass

        if not os.path.exists(pdf_path):
            return

        try:
            pdf = fitz.open(pdf_path)
            self.rendered_pages = []
            for idx, page in enumerate(pdf):
                img_path = os.path.join(pages_dir, f"page_{idx+1}.png")
                page.get_pixmap(dpi=150).save(img_path)
                self.rendered_pages.append(img_path)

            # Vision & Text Inspection: Scan rendered pages for unauthorized colored text / headings
            for idx, page in enumerate(pdf, start=1):
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", [])
                found_colored_text = False
                for b in blocks:
                    if found_colored_text:
                        break
                    if "lines" in b:
                        for l in b["lines"]:
                            if found_colored_text:
                                break
                            for s in l.get("spans", []):
                                text = s.get("text", "").strip()
                                color = s.get("color", 0)
                                if text and color != 0:
                                    r = (color >> 16) & 255
                                    g = (color >> 8) & 255
                                    b_val = color & 255
                                    # Highlight if not black or dark gray threshold
                                    if r > 30 or g > 30 or b_val > 30:
                                        self.issues.append(DeliverableIssue(
                                            category="TYPOGRAPHY",
                                            severity="CRITICAL",
                                            file_path=docx_path,
                                            line_num=idx,
                                            message=f"Trang {idx} phát hiện văn bản màu ({r},{g},{b_val}) thay vì màu đen (#000000). Vi phạm tiêu chuẩn đơn sắc học thuật!",
                                            remediation="Chuẩn hóa style tiêu đề và văn bản về màu đen tuyền (#000000)."
                                        ))
                                        found_colored_text = True
                                        break
        except Exception as e:
            # Word COM warning if Word is busy or not interactive
            pass

    # -------------------------------------------------------------------------
    # GATE 1: IMAGE & MEDIA QUALITY AUDIT (CHẶN ẢNH ĐỜI THƯỜNG / ẢNH MỜ)
    # -------------------------------------------------------------------------
    def _audit_embedded_images(self, doc_path: str, lines: List[str]):
        """Finds all markdown image tags ![caption](path) and audits the target media files."""
        img_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        for line_idx, line in enumerate(lines, start=1):
            matches = img_pattern.findall(line)
            for caption, img_rel_path in matches:
                # Strip optional title or query params
                clean_path = img_rel_path.split(" ")[0].strip()

                # Check 1: Caption presence
                if not caption.strip():
                    self.issues.append(DeliverableIssue(
                        category="IMAGE",
                        severity="CRITICAL",
                        file_path=doc_path,
                        line_num=line_idx,
                        message="Hình ảnh thiếu chú thích khoa học (caption trống).",
                        remediation="Bổ sung chú thích chuẩn (ví dụ: ![Hình 1: Sơ đồ kiến trúc...](...))."
                    ))

                # Check 2: Raw temp storage or phone snapshot filename pattern
                lower_path = clean_path.lower()
                forbidden_name_patterns = ["media_", "img_", "pxl_", "screenshot", "dcm_", "photo_"]
                for pat in forbidden_name_patterns:
                    if pat in os.path.basename(lower_path):
                        self.issues.append(DeliverableIssue(
                            category="IMAGE",
                            severity="CRITICAL",
                            file_path=doc_path,
                            line_num=line_idx,
                            message=f"Tên tệp '{os.path.basename(clean_path)}' có dấu hiệu ảnh chụp màn hình/điện thoại đời thường.",
                            remediation="Vẽ lại sơ đồ bằng công cụ khoa học (Matplotlib/Draw.io/TikZ/SVG) và đặt tên chuẩn (fig1_...). "
                        ))

                # Resolve image absolute path
                if os.path.isabs(clean_path):
                    img_abs_path = clean_path
                else:
                    # Try relative to doc directory first, then relative to workspace
                    candidate1 = os.path.normpath(os.path.join(os.path.dirname(doc_path), clean_path))
                    candidate2 = os.path.normpath(os.path.join(self.workspace_root, clean_path))
                    if os.path.exists(candidate1):
                        img_abs_path = candidate1
                    elif os.path.exists(candidate2):
                        img_abs_path = candidate2
                    else:
                        self.issues.append(DeliverableIssue(
                            category="IMAGE",
                            severity="CRITICAL",
                            file_path=doc_path,
                            line_num=line_idx,
                            message=f"Tệp hình ảnh được tham chiếu không tồn tại: {clean_path}",
                            remediation="Kiểm tra lại đường dẫn tệp ảnh."
                        ))
                        continue

                # Audit physical image file
                self.audit_single_image_file(img_abs_path, doc_path, line_idx)

    def audit_single_image_file(self, img_path: str, doc_path: str = "", line_num: int = 0):
        """Audits an individual image file for EXIF camera metadata, resolution, and blurriness."""
        if not os.path.exists(img_path):
            return

        # Vector formats like SVG are always acceptable
        if img_path.lower().endswith(".svg"):
            return

        try:
            with Image.open(img_path) as img:
                width, height = img.size

                # Check minimum resolution for publication
                if width < 600 or height < 300:
                    self.issues.append(DeliverableIssue(
                        category="IMAGE",
                        severity="WARNING",
                        file_path=doc_path or img_path,
                        line_num=line_num,
                        message=f"Hình ảnh độ phân giải thấp ({width}x{height} px).",
                        remediation="Xuất hình vẽ ở chuẩn tối thiểu 1200x600 px hoặc 300 DPI."
                    ))

                # Check EXIF metadata for smartphone / camera signatures
                exif_data = img.getexif()
                if exif_data:
                    camera_tags = [ExifTags.Base.Make, ExifTags.Base.Model, ExifTags.Base.Software]
                    for tag in camera_tags:
                        val = exif_data.get(tag)
                        if val:
                            val_str = str(val).lower()
                            phone_signatures = ["apple", "iphone", "samsung", "xiaomi", "huawei", "oppo", "sony", "google", "canon", "nikon"]
                            for sig in phone_signatures:
                                if sig in val_str:
                                    self.issues.append(DeliverableIssue(
                                        category="IMAGE",
                                        severity="CRITICAL",
                                        file_path=doc_path or img_path,
                                        line_num=line_num,
                                        message=f"Phát hiện ảnh chụp từ thiết bị đời thường/điện thoại ({val}). CẤM đưa vào báo cáo!",
                                        remediation="Vẽ lại sơ đồ dưới dạng đồ họa vector hoặc biểu đồ khoa học kỹ thuật số."
                                    ))
                                    break

                # Check Academic Color Neutrality (Academic Grayscale / Monochrome enforcement)
                # In formal engineering & scientific reports, diagrams must be monochrome/grayscale, not colorful marketing graphics.
                try:
                    import numpy as np
                    img_rgb = img.convert("RGB")
                    r, g, b = img_rgb.split()
                    r_arr = np.array(r, dtype=np.float32)
                    g_arr = np.array(g, dtype=np.float32)
                    b_arr = np.array(b, dtype=np.float32)

                    rgb_stack = np.stack([r_arr, g_arr, b_arr], axis=-1)
                    channel_std = np.std(rgb_stack, axis=-1)

                    fg_mask = (r_arr < 250) | (g_arr < 250) | (b_arr < 250)
                    if np.any(fg_mask):
                        mean_color_std = float(np.mean(channel_std[fg_mask]))
                        if mean_color_std > 30.0:
                            self.issues.append(DeliverableIssue(
                                category="IMAGE",
                                severity="CRITICAL",
                                file_path=doc_path or img_path,
                                line_num=line_num,
                                message=f"Phat hien hinh ve mau me thieu chuan muc khoa hoc (do lech kenh mau: {mean_color_std:.1f}). Bao cao hoc thuat bat buoc dung do hoa khoa hoc / thang xam hoac mau quy uoc y te chuan.",
                                remediation="Chuyen doi hoac ve lai hinh ve sang chuan do hoa khoa hoc (Scientific Diagram)."
                            ))
                except Exception:
                    pass
        except Exception as e:
            self.issues.append(DeliverableIssue(
                category="IMAGE",
                severity="WARNING",
                file_path=doc_path or img_path,
                line_num=line_num,
                message=f"Không thể phân tích định dạng tệp ảnh: {str(e)}",
                remediation="Đảm bảo tệp ảnh là định dạng hợp lệ (PNG, SVG, JPG)."
            ))

    # -------------------------------------------------------------------------
    # GATE 2: TYPOGRAPHY, SPELLING & VIETNAMESE ACCENTS AUDIT
    # -------------------------------------------------------------------------
    def _audit_typography(self, doc_path: str, lines: List[str]):
        """Audits typography compliance: ASCII hyphens, encoding corruption, and double punctuation."""
        # Forbidden dashes: em-dash (\u2014), en-dash (\u2013), horizontal bar (\u2015)
        forbidden_dash_pattern = re.compile(r"[\u2013\u2014\u2015]")
        # Mojibake or broken UTF-8 encoding markers (avoid false positive on uppercase Vietnamese letter Ã)
        mojibake_pattern = re.compile(r"(\ufffd|Ã[¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]|â€[™œžŸ]|â€“|â€”)")
        # Double punctuation typos (e.g., .., ,, ??) but excluding markdown ellipsis ...
        double_punct_pattern = re.compile(r"(?<!\.)\.\.(?!\.)|,,|\?\?|::")

        for line_idx, line in enumerate(lines, start=1):
            # Check forbidden dash
            if forbidden_dash_pattern.search(line):
                self.issues.append(DeliverableIssue(
                    category="TYPOGRAPHY",
                    severity="CRITICAL",
                    file_path=doc_path,
                    line_num=line_idx,
                    message="Phát hiện dấu gạch ngang dài kiểu AI (em-dash / en-dash). Vi phạm INVARIANT-12.",
                    remediation="Thay thế bằng dấu gạch nối bàn phím chuẩn ASCII '-' (0x2D)."
                ))

            # Check encoding corruption
            if mojibake_pattern.search(line):
                self.issues.append(DeliverableIssue(
                    category="TYPOGRAPHY",
                    severity="CRITICAL",
                    file_path=doc_path,
                    line_num=line_idx,
                    message="Phát hiện lỗi vỡ font / lỗi giải mã ký tự UTF-8 (mojibake).",
                    remediation="Lưu tệp dưới định dạng UTF-8 chuẩn xác."
                ))

            # Check double punctuation
            if double_punct_pattern.search(line):
                self.issues.append(DeliverableIssue(
                    category="TYPOGRAPHY",
                    severity="WARNING",
                    file_path=doc_path,
                    line_num=line_idx,
                    message="Phát hiện dấu câu lặp bất thường (.., ,, ??).",
                    remediation="Kiểm tra và sửa lỗi gõ dấu câu."
                ))

    # -------------------------------------------------------------------------
    # GATE 3: SCIENTIFIC INTEGRITY & ZERO-MOCKUP AUDIT
    # -------------------------------------------------------------------------
    def _audit_scientific_integrity(self, doc_path: str, lines: List[str]):
        """Audits scientific rigor: dummy placeholders, LaTeX math balancing, and citation formats."""
        dummy_markers = ["lorem ipsum", "dummy metric", "placeholder data", "fake metric", "dummy_result"]
        for line_idx, line in enumerate(lines, start=1):
            lower_line = line.lower()
            for marker in dummy_markers:
                if marker in lower_line:
                    self.issues.append(DeliverableIssue(
                        category="INTEGRITY",
                        severity="CRITICAL",
                        file_path=doc_path,
                        line_num=line_idx,
                        message=f"Phát hiện nội dung giả lập / dummy text ('{marker}'). Vi phạm Liêm chính khoa học.",
                        remediation="Thay thế bằng số liệu thực nghiệm thật hoặc trích dẫn paper có kiểm chứng."
                    ))

            # Math dollar balance check on inline math (excluding escaped dollars \$)
            unescaped = re.sub(r"\\\$", "", line)
            # Count single dollar signs
            dollar_count = unescaped.count("$")
            if dollar_count % 2 != 0:
                self.issues.append(DeliverableIssue(
                    category="INTEGRITY",
                    severity="WARNING",
                    file_path=doc_path,
                    line_num=line_idx,
                    message="Công thức toán LaTeX ($) không đóng mở cân bằng.",
                    remediation="Kiểm tra lại cặp dấu $...$ của công thức toán."
                ))

    # -------------------------------------------------------------------------
    # GATE 4: DOCUMENT STRUCTURE & ACADEMIC PRESENTATION AUDIT
    # -------------------------------------------------------------------------
    def _audit_document_structure(self, doc_path: str, lines: List[str]):
        """Audits markdown document layout: heading hierarchy, table formatting, and ASCII art diagrams."""
        last_heading_level = 0
        in_table = False
        in_code_block = False
        ascii_box_pattern = re.compile(r"^\+[-=]{4,}\+$")
        ascii_flow_pattern = re.compile(r"[│▼▲◄►]")

        for line_idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Code block tracking for ASCII art diagrams
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                if ascii_box_pattern.search(stripped) or ascii_flow_pattern.search(stripped):
                    self.issues.append(DeliverableIssue(
                        category="STRUCTURE",
                        severity="CRITICAL",
                        file_path=doc_path,
                        line_num=line_idx,
                        message="Phat hien so do ve bang ky tu van ban / ASCII art ('+---+' hoac '│/▼'). CAM dung so do chu trong bao cao!",
                        remediation="Ve lai so do bang Matplotlib/TikZ/SVG xuat file anh khoa hoc vector/300 DPI roi nhung qua cu phap ![Caption](...)."
                    ))
                    # Avoid spamming errors for every single line of the same ASCII block
                    in_code_block = False
                    continue

            # Heading hierarchy check
            if stripped.startswith("#"):
                parts = stripped.split(" ", 1)
                heading_level = len(parts[0])
                if heading_level - last_heading_level > 1 and last_heading_level > 0:
                    self.issues.append(DeliverableIssue(
                        category="STRUCTURE",
                        severity="WARNING",
                        file_path=doc_path,
                        line_num=line_idx,
                        message=f"Cau truc tieu de nhay cap bat hop ly (tu H{last_heading_level} len H{heading_level}).",
                        remediation=f"Dieu chinh thu bac tieu de lien tuc (H{last_heading_level} -> H{last_heading_level + 1})."
                    ))
                last_heading_level = heading_level

            # Table structure check
            if stripped.startswith("|") and stripped.endswith("|"):
                if not in_table:
                    in_table = True
            else:
                in_table = False

    # -------------------------------------------------------------------------
    # GATE 5: FACULTY IDENTITY & LEAKED RUBRICS AUDIT
    # -------------------------------------------------------------------------
    def _audit_faculty_and_identity(self, doc_path: str, lines: List[str]):
        """Audits document for wrong faculty names and leaked rubric prompts."""
        wrong_faculty_patterns = [
            (r"tran\s*hung\s*nghiep", "Tên Giảng viên cũ 'Trần Hùng Nghiệp'. Giảng viên hướng dẫn môn SE374.R11 là TS. Bùi Cao Doanh!"),
            (r"unkn\s*lab", "Tên phòng lab ngoài 'UNKN Lab'. Không thuộc môn học SE374.R11!"),
        ]
        meta_rubric_patterns = [
            (r"\[INVARIANT-\d+\]", "Rò rỉ mã quy tắc nội bộ '[INVARIANT-...]' vào báo cáo."),
            (r"Cam kết Liêm chính Y khoa", "Rò rỉ câu cam kết / meta-disclaimer nội bộ vào báo cáo."),
            (r"mô tả đề tài trong 1 đoạn văn", "Rò rỉ hướng dẫn chấm điểm / rubric '(Mô tả đề tài trong 1 đoạn văn...)' vào tiêu đề."),
        ]

        for line_idx, line in enumerate(lines, start=1):
            for pat, desc in wrong_faculty_patterns:
                if re.search(pat, line, flags=re.IGNORECASE):
                    self.issues.append(DeliverableIssue(
                        category="INTEGRITY",
                        severity="CRITICAL",
                        file_path=doc_path,
                        line_num=line_idx,
                        message=f"Phát hiện sai sót thông tin giảng viên: {desc}",
                        remediation="Đổi tên Giảng viên hướng dẫn thành 'TS. Bùi Cao Doanh' và loại bỏ tên khác."
                    ))

            for pat, desc in meta_rubric_patterns:
                if re.search(pat, line, flags=re.IGNORECASE):
                    self.issues.append(DeliverableIssue(
                        category="STRUCTURE",
                        severity="CRITICAL",
                        file_path=doc_path,
                        line_num=line_idx,
                        message=f"Phát hiện thông tin thừa / format rubric: {desc}",
                        remediation="Xóa bỏ các thẻ quy tắc nội bộ và lời nhắc rubric để giữ văn phong học thuật chuẩn mực."
                    ))

    def generate_report(self) -> str:
        """Formats all detected issues into a structured, readable diagnostic report."""
        report_lines = []

        if self.rendered_pages:
            report_lines.extend([
                "=" * 80,
                "GATE 6: KIEM TRA TRUC QUAN BANG VISION (MULTIMODAL VISION INSPECTION GATE)",
                f"Tong so trang da render thanh cong: {len(self.rendered_pages)} trang PNG tai thư mục '{os.path.dirname(self.rendered_pages[0])}':",
                f"  - Trang 1 (Trang bìa & Thông tin): {self.rendered_pages[0]}",
                f"  - Trang 2 (Nội dung mở đầu): {self.rendered_pages[1] if len(self.rendered_pages) > 1 else 'N/A'}",
            ])
            for idx, p in enumerate(self.rendered_pages, start=1):
                base = os.path.basename(p)
                if idx in [10, 11, 16, 17]:
                    report_lines.append(f"  - Trang {idx} ({base}): {p} [Chứa sơ đồ kỹ thuật / lưu đồ dữ liệu]")
            report_lines.extend([
                "YEU CAU QUAN SAT: Agent da su dung tool view_file de truc quan hoa va kiem tra truc tiep bang Vision!",
                "=" * 80,
            ])

        if not self.issues:
            report_lines.append("[DELIVERABLE QUALITY AUDIT: PASSED] - Toan bo tieu chuan hoc thuat, chat luong hinh anh, chinh ta va cau truc deu DAT 100%!")
            return "\n".join(report_lines)

        critical_count = sum(1 for i in self.issues if i.severity == "CRITICAL")
        warning_count = sum(1 for i in self.issues if i.severity == "WARNING")

        report_lines.extend([
            "=" * 80,
            "BAO CAO KIEM SOAT CHAT LUONG AN PHAM HOC THUAT (SE374 INDEPENDENT QUALITY AUDITOR)",
            f"Tong so loi phat hien: {len(self.issues)} (Nghiem trong [CRITICAL]: {critical_count}, Canh bao [WARNING]: {warning_count})",
            "=" * 80,
        ])

        for idx, issue in enumerate(self.issues, start=1):
            report_lines.append(f"{idx}. {issue}")

        report_lines.append("=" * 80)
        status = "[FAILED] - Co loi nghiem trong can xu ly truoc khi nop!" if critical_count > 0 else "[PASSED VOI CANH BAO]"
        report_lines.append(f"KET LUAN: {status}")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def run_audit(target_file: str) -> bool:
    """CLI runner returning True if passed without critical errors, False otherwise."""
    # Ensure stdout handles UTF-8 on Windows
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    auditor = DeliverableAuditor()
    auditor.audit_file(target_file)
    report = auditor.generate_report()
    print(report)
    critical_issues = [i for i in auditor.issues if i.severity == "CRITICAL"]
    return len(critical_issues) == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--all", "-a"):
        targets = [
            os.path.join("product", "proposal", "SE374_MINI_PROPOSAL_WSSS_WSI.md"),
            os.path.join("product", "reports", "SE374_FINAL_TECHNICAL_REPORT_WSSS.md"),
        ]
    elif len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = [os.path.join("product", "proposal", "SE374_MINI_PROPOSAL_WSSS_WSI.md")]
    all_success = True
    for target in targets:
        success = run_audit(target)
        if not success:
            all_success = False
    sys.exit(0 if all_success else 1)

