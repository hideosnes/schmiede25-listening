import argparse
import importlib
import pkgutil
import sys
from typing import Dict

def discover_processes() -> Dict[str, object]:
    processes = {}
    try:
        import processes as processes_pkg # todo: pkg must exist
    except Exception as e:
        print("Error importing 'processes':", e)
        return processes

    for finder, name, ispkg in pkgutil.iter_modules(processes_pkg.__path__):
        module_name = f"processes.{name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"Failed to import {module_name}: {e}")
            continue

        # expecting each module to expose: NAME (str), add_arguments(parser), run(args)
        mod_name = getattr(module, "NAME", name)
        if hasattr(module, "run") and hasattr(module, "add_arguments"):
            processes[mod_name] = module
    return processes

def build_cli(processes_map: Dict[str, object]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project runner - discovers and runs processes in the processes/ folder"
    )
    parser.add_argument("--list", action="store_true", help="List available processes")
    subparsers = parser.add_subparsers(dest="process")

    for name, module in processes_map.items():
        sp = subparsers.add_parser(name, help=getattr(module, "DESCRIPTIOM", None))
        module.add_arguments(sp)

    args = parser.parse_args()
    return args

def main():
    processes_map = discover_processes()
    args = build_cli(processes_map)

    if getattr(args, "list", False) or args.process is None:
        if processes_map:
            print("Available process:")
            for n in sorted(processes_map):
                desc = getattr(processes_map[n], "DESCRIPTION", "")
                print(f" - {n}" + (f": {desc}" if desc else ""))
            print("\nUsage examples:")
            print(" python app.py --list")
            print(" python app.py <process-name> --help")
        else:
            print("No processes found in the 'processes'- folder")
        return

    module = processes_map.get(args.process)
    if module is None:
        print(f"Unknown process '{args.process}'")
        return
    try:
        module.run(args)
    except Exception as e:
        print(f"Error while running process '{args.process}': {e}")
        raise

if __name__ == "__main__":
    main()
