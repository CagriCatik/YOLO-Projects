import zipfile
from pathlib import Path
from .log import log

def zip_run(run_dir: str, name: str = "my_model", out_dir: str = "artifacts"):
    run_dir = Path(run_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"best.pt not found under {weights.parent}")
    pkg_dir = out_dir / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    # Copy files (do not duplicate heavy images; keep the training curve png and results)
    (pkg_dir / "weights").mkdir(exist_ok=True)
    import shutil
    shutil.copy2(weights, pkg_dir / "weights" / f"{name}.pt")
    # Copy minimal run metadata
    for fn in ["results.png", "results.csv", "hyp.yaml", "opt.yaml", "args.yaml"]:
        f = run_dir / fn
        if f.exists():
            shutil.copy2(f, pkg_dir / fn)
    # Zip
    zip_path = out_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pkg_dir.rglob("*"):
            zf.write(p, p.relative_to(out_dir))
    log(f"Wrote {zip_path}")
    return str(zip_path)
