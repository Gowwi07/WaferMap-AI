import os
import glob
import matplotlib.pyplot as plt

os.makedirs("frontend/samples", exist_ok=True)
plot_files = glob.glob("frontend/samples/plot_*.py")

print(f"Found {len(plot_files)} plot scripts. Generating images...")

for f in sorted(plot_files):
    try:
        with open(f, "r", encoding="utf-8") as file:
            code = file.read()
            
        base_name = os.path.basename(f).replace(".py", ".png")
        out_path = f"frontend/samples/{base_name}"
        
        # Append logic to save the matplotlib figure
        runner_code = code + f"\nimport matplotlib.pyplot as plt\nplt.savefig('{out_path}', dpi=150, bbox_inches='tight')\nplt.close('all')\n"
        
        # Execute the script in a fresh local dictionary
        namespace = {}
        exec(runner_code, namespace)
        print(f"Generated {out_path}")
        
    except Exception as e:
        print(f"Failed to generate {f}: {e}")
