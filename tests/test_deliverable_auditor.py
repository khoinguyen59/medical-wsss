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
        proposal_path = os.path.join("product", "proposal", "SE374_MINI_PROPOSAL_WSSS_WSI.md")
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

    def test_valid_docx_proposal_passes_audit(self, tmp_path):
        """Compiled DOCX proposal must pass audit with 0 critical issues."""
        auditor = DeliverableAuditor()
        docx_path = os.path.join("product", "proposal", "SE374_MINI_PROPOSAL_WSSS_TISSUE.docx")
        assert os.path.exists(docx_path), "DOCX proposal file must exist"

        issues = auditor.audit_docx_file(docx_path, preview_output_dir=str(tmp_path))
        critical_issues = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical_issues) == 0, f"DOCX has critical issues: {[str(i) for i in critical_issues]}"

    def test_ignores_bash_and_python_comments_inside_fenced_code_blocks(self):
        """Auditor must NOT misclassify '# comment' inside ```bash or ~~~python fences as H1 headings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = os.path.join(tmpdir, "code_fence_report.md")
            content = (
                "# Chuong 1: Gioi thieu\n\n"
                "## 1.1 Cai dat moi truong\n\n"
                "```bash\n"
                "# Day la comment trong bash, KHONG phai H1\n"
                "python -m pytest tests/\n"
                "```\n\n"
                "~~~python\n"
                "# Day la comment trong khối tilde fence\n"
                "x = 42\n"
                "~~~\n\n"
                "### 1.1.1 Chi tiet thuc thi\n"
            )
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(content)

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(md_file)
            heading_jump_issues = [
                i for i in issues if i.category == "STRUCTURE" and "nhay cap" in i.message
            ]
            assert len(heading_jump_issues) == 0, (
                f"Comments inside code blocks caused false heading-jump issues: {heading_jump_issues}"
            )

    def test_ascii_art_inside_code_block_does_not_corrupt_subsequent_fence_state(self):
        """When an ASCII art box is caught inside a code block, closing fence must still close cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = os.path.join(tmpdir, "ascii_then_valid.md")
            content = (
                "# Chuong 1\n\n"
                "## 1.1 So do loi\n\n"
                "```\n"
                "+----------------+\n"
                "| BOX 1          |\n"
                "+----------------+\n"
                "```\n\n"
                "### 1.1.1 Muc con hop le\n\n"
                "```bash\n"
                "# Comment hop le sau khoi ASCII\n"
                "echo ok\n"
                "```\n\n"
                "#### 1.1.1.1 Muc chau hop le\n"
            )
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(content)

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(md_file)
            ascii_issues = [i for i in issues if i.category == "STRUCTURE" and "ASCII art" in i.message]
            heading_jump_issues = [
                i for i in issues if i.category == "STRUCTURE" and "nhay cap" in i.message
            ]
            assert len(ascii_issues) == 1
            assert len(heading_jump_issues) == 0

    def test_html_img_tags_detect_missing_empty_alt_low_res_and_allow_valid_and_remote(self):
        """Auditor must inspect HTML <img> tags (including multiline), skip code fences, and validate assets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Valid 300-DPI monochrome diagram (800x400)
            mono_path = os.path.join(tmpdir, "fig_arch.png")
            Image.new("L", (800, 400), color=255).save(mono_path, dpi=(300, 300))

            # 2. Valid 300-DPI >= 1800px non-diagram medical chart with color
            sci_path = os.path.join(tmpdir, "ablation_chart.png")
            sci_img = Image.new("RGB", (1800, 900), color=(255, 255, 255))
            for x in range(100, 500):
                for y in range(100, 500):
                    sci_img.putpixel((x, y), (220, 40, 40))
            sci_img.save(sci_path, dpi=(300, 300))

            # 3. Low-res image (200x100)
            tiny_path = os.path.join(tmpdir, "tiny.png")
            Image.new("L", (200, 100), color=180).save(tiny_path)

            # 4. Low-DPI colorful architecture diagram
            bad_diag_path = os.path.join(tmpdir, "colorful_diagram.png")
            diag_img = Image.new("RGB", (800, 400), color=(255, 255, 255))
            for x in range(100, 300):
                for y in range(100, 300):
                    diag_img.putpixel((x, y), (255, 20, 147))
            diag_img.save(bad_diag_path)

            md_file = os.path.join(tmpdir, "html_imgs.md")
            content = (
                "# Bao cao\n\n"
                '<img\n  src="fig_arch.png"\n  alt="Hinh 1: Kien truc don sac"\n/>\n\n'
                '<img src="ablation_chart.png" alt="Bieu do doi sanh 300 DPI"/>\n\n'
                '<img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C" alt="PyTorch"/>\n\n'
                '<img src="missing_fig.png" alt="Missing figure"/>\n\n'
                "<img src='tiny.png' alt=''/>\n\n"
                '<img src="colorful_diagram.png" alt="So do kien truc mau me"/>\n\n'
                '<img alt="No src attribute"/>\n\n'
                "```html\n"
                '<img src="missing_inside_fence.png" alt=""/>\n'
                "```\n"
            )
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(content)

            auditor = DeliverableAuditor(workspace_root=tmpdir)
            issues = auditor.audit_markdown_file(md_file)

            msgs = [i.message for i in issues]
            assert any("missing_fig.png" in m for m in msgs)
            assert any("thiếu chú thích khoa học" in m for m in msgs)
            assert any("độ phân giải thấp" in m for m in msgs)
            assert any("mau me" in m for m in msgs)
            assert any("thiếu thuộc tính src" in m for m in msgs)
            assert not any("missing_inside_fence.png" in m for m in msgs)
            assert not any("fig_arch.png" in m for m in msgs)
            assert not any("ablation_chart.png" in m for m in msgs)

    def test_readme_passes_deliverable_auditor(self):
        """Repository README.md must pass DeliverableAuditor with 0 CRITICAL issues."""
        auditor = DeliverableAuditor()
        issues = auditor.audit_markdown_file("README.md")
        critical_issues = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical_issues) == 0, f"README.md critical issues: {[str(i) for i in critical_issues]}"



