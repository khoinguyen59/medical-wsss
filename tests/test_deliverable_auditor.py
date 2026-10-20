"""
Unit tests for DeliverableAuditor (Independent Quality Assurance System).
Tests all 4 Quality Gates:
1. Image & Visual Quality Gate (Blocks raw phone snapshots, empty captions, low-res)
2. Typography & Spelling Gate (Blocks em-dash, en-dash, mojibake)
3. Scientific Integrity Gate (Blocks dummy metrics, lorem ipsum)
4. Document Structure Gate (Heading hierarchy)
"""

import os
import tempfile
import pytest
from PIL import Image
from src.utils.deliverable_auditor import DeliverableAuditor


class TestDeliverableAuditor:

    def test_valid_proposal_passes_audit(self):
        """Current proposal should pass audit with 0 critical issues."""
        auditor = DeliverableAuditor()
        proposal_path = os.path.join("product", "proposal", "SE374_MINI_PROPOSAL_WSSS_TISSUE.md")
        assert os.path.exists(proposal_path), "Proposal file must exist"

        issues = auditor.audit_markdown_file(proposal_path)
        critical_issues = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical_issues) == 0, f"Proposal has critical issues: {[str(i) for i in critical_issues]}"

    def test_blocks_raw_phone_snapshot_name(self):
        """Auditor must flag files with camera/phone snapshot patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = os.path.join(tmpdir, "bad_report.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write("# Báo cáo\n\n![Sơ đồ](media_123456789.png)\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            # Should catch raw snapshot pattern
            image_issues = [i for i in issues if i.category == "IMAGE" and i.severity == "CRITICAL"]
            assert len(image_issues) >= 1
            assert any("ảnh chụp màn hình/điện thoại" in i.message for i in image_issues)

    def test_blocks_empty_caption(self):
        """Auditor must flag images without descriptive scientific captions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = os.path.join(tmpdir, "bad_report.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write("# Báo cáo\n\n![](fig1.png)\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            caption_issues = [i for i in issues if "thiếu chú thích khoa học" in i.message]
            assert len(caption_issues) == 1

    def test_blocks_forbidden_em_dash_and_en_dash(self):
        """Auditor must enforce ASCII hyphen and block em-dash (\\u2014) and en-dash (\\u2013)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = os.path.join(tmpdir, "bad_typography.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write("# Báo cáo \u2014 Vi phạm\n\nKhoảng cách 10 \u2013 20px.\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            dash_issues = [i for i in issues if i.category == "TYPOGRAPHY" and "gạch ngang dài" in i.message]
            assert len(dash_issues) >= 1

    def test_blocks_dummy_metrics_and_mockup(self):
        """Auditor must catch lorem ipsum and dummy data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = os.path.join(tmpdir, "fake_report.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write("# Kết quả\n\nLorem ipsum dolor sit amet. dummy metric = 0.99\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            integrity_issues = [i for i in issues if i.category == "INTEGRITY"]
            assert len(integrity_issues) >= 1

    def test_flags_low_resolution_images(self):
        """Auditor should warn if an embedded image is low resolution (<600x300)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            low_res_img = os.path.join(tmpdir, "tiny_fig.png")
            img = Image.new("RGB", (200, 100), color="blue")
            img.save(low_res_img)

            bad_md = os.path.join(tmpdir, "report.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write(f"# Báo cáo\n\n![Hình 1: Test]({low_res_img})\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            low_res_issues = [i for i in issues if "độ phân giải thấp" in i.message]
            assert len(low_res_issues) >= 1

    def test_blocks_ascii_art_diagrams(self):
        """Auditor must catch ASCII art boxes or text-based flowcharts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = os.path.join(tmpdir, "ascii_report.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write("# Sơ đồ\n\n```\n+-----------------------+\n| TANG 1: CLIENT        |\n+-----------------------+\n```\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            ascii_issues = [i for i in issues if i.category == "STRUCTURE" and "ASCII art" in i.message]
            assert len(ascii_issues) >= 1

    def test_blocks_colorful_figures_and_enforces_monochrome(self):
        """Auditor must catch overly colorful/pastel figures and enforce academic monochrome."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create a colorful figure (bright red, green, blue patches)
            colorful_img_path = os.path.join(tmpdir, "colorful_diagram.png")
            img = Image.new("RGB", (800, 400), color=(255, 255, 255))
            # Draw strong color blocks
            for x in range(100, 300):
                for y in range(100, 300):
                    img.putpixel((x, y), (255, 20, 147))  # Deep pink
            for x in range(400, 600):
                for y in range(100, 300):
                    img.putpixel((x, y), (0, 200, 50))    # Bright green
            img.save(colorful_img_path)

            bad_md = os.path.join(tmpdir, "colorful_report.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write(f"# Báo cáo\n\n![Hình 1: Sơ đồ kiến trúc]({colorful_img_path})\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            color_issues = [i for i in issues if i.category == "IMAGE" and "mau me" in i.message]
            assert len(color_issues) >= 1

    def test_blocks_wrong_faculty_and_external_lab(self):
        """Auditor must catch wrong faculty name 'Tran Hung Nghiep' or 'UNKN Lab'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = os.path.join(tmpdir, "bad_faculty.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write("# Báo cáo\n\n- GVHD: TS. Trần Hùng Nghiệp\n- Lab: UNKN Lab\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            faculty_issues = [i for i in issues if i.category == "INTEGRITY" and ("giảng viên" in i.message.lower() or "giang vien" in i.message.lower())]
            assert len(faculty_issues) >= 1

    def test_blocks_leaked_rubrics_and_invariants(self):
        """Auditor must catch leaked rubric prompts, invariant tokens, and internal disclaimers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = os.path.join(tmpdir, "bad_rubric.md")
            with open(bad_md, "w", encoding="utf-8") as f:
                f.write("### 1.1 Project Description (Mô tả đề tài trong 1 đoạn văn, \\le 150 từ)\n[INVARIANT-08]\n*(Cam kết Liêm chính Y khoa...)*\n")

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(bad_md)

            rubric_issues = [i for i in issues if i.category == "STRUCTURE" and "thông tin thừa" in i.message.lower()]
            assert len(rubric_issues) >= 3

    def test_valid_docx_proposal_passes_audit(self):
        """Compiled DOCX proposal must pass audit with 0 critical issues."""
        auditor = DeliverableAuditor()
        docx_path = os.path.join("product", "proposal", "SE374_MINI_PROPOSAL_WSSS_TISSUE.docx")
        assert os.path.exists(docx_path), "DOCX proposal file must exist"

        issues = auditor.audit_docx_file(docx_path)
        critical_issues = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical_issues) == 0, f"DOCX has critical issues: {[str(i) for i in critical_issues]}"

