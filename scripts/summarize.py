"""Pool per-seed outcomes of evaluation cells: python summarize.py <eval dir> "<glob of cell names>"."""
import glob, json, math, os, sys

root, pattern = sys.argv[1], sys.argv[2]
k = n = 0
for f in sorted(glob.glob(os.path.join(root, pattern, "summary.json"))):
    per = json.load(open(f))["per_seed"]
    k += sum(bool(e["success"]) for e in per); n += len(per)
if not n:
    sys.exit(f"no cells match {pattern}")
p, z = k / n, 1.96
c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
print(f"{pattern}: {k}/{n} = {100 * p:.1f}%  (95% CI {100 * (c - h):.1f}-{100 * (c + h):.1f})")
