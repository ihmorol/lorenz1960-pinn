"""Verify frozen manuscript artifacts without TeX, training or dependencies."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "writings/iceeict2027-final"
TEXT = {".tex", ".bib", ".bbl", ".cls", ".bst", ".py", ".json", ".txt", ".md", ".csv", ".svg", ".diff", ".html", ".js"}
UPLOAD = {
    "main.tex", "main.pdf", "manuscript.txt", "references.bib", "main.bbl",
    "IEEEtran.cls", "IEEEtran.bst", "ablation-table.tex", "figures/method.pdf",
    "figures/plateau-results.pdf", "figures/solution_vs_reference.pdf",
}


def same_source(name, left, right):
    # Git may normalize text line endings; frozen ZIP hashes remain byte-exact.
    if Path(name).suffix in TEXT:
        return left.replace(b"\r\n", b"\n") == right.replace(b"\r\n", b"\n")
    return left == right


def main():
    manifest = json.loads((PAPER / "delivery-manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(ROOT / "writings/iceeict2027-final.zip") as archive:
        expected = {"iceeict2027-final/" + p for p in manifest["files"]}
        expected.add("iceeict2027-final/delivery-manifest.json")
        assert set(archive.namelist()) == expected and len(archive.namelist()) == len(expected)
        assert archive.testzip() is None
        assert json.loads(archive.read("iceeict2027-final/delivery-manifest.json")) == manifest
        for name, record in manifest["files"].items():
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            data = archive.read("iceeict2027-final/" + name)
            assert len(data) == record["bytes"], name
            assert hashlib.sha256(data).hexdigest() == record["sha256"], name
            assert same_source(name, data, (PAPER / name).read_bytes()), name
    pdf = (PAPER / "main.pdf").read_bytes()
    assert len(pdf) < 3_000_000
    assert hashlib.sha256(pdf).hexdigest() == manifest["pdf_sha256"]
    for name in ["independent-review.json", "final-verification.json"]:
        report = json.loads((PAPER / "review" / name).read_text(encoding="utf-8"))
        assert report["pdf_sha256"] == manifest["pdf_sha256"]
        assert report["paper_source_commit"] == manifest["paper_source_commit"]
        assert report["pages"] == 6 and report["all_fonts_embedded"]
    with zipfile.ZipFile(ROOT / "writings/iceeict2027-manuscript.zip") as archive:
        assert set(archive.namelist()) == UPLOAD and len(archive.namelist()) == len(UPLOAD)
        assert archive.testzip() is None
        for name in UPLOAD:
            assert same_source(name, archive.read(name), (PAPER / name).read_bytes()), name
    print(f"Verified {len(manifest['files'])} frozen files and both ZIPs; audited PDF unchanged.")


if __name__ == "__main__":
    main()
