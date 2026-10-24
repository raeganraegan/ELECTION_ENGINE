import argparse
import ast
from INANNA_V2 import main as INANNA
from ENKI_V2 import main as ENKI

def main():
    parser = argparse.ArgumentParser(description="Master Engine Runner")
    
    # Shared Core Arguments (Optional Flags shared to both)
    parser.add_argument("--seat_name", type=str, default=None, help="Name of the seat")
    parser.add_argument("--output_root", type=str, default=None, help="Name of where the output root is")

    # ENKI Arguments (Optional Flags)
    parser.add_argument("--booth_data", type=str, default=None, help="Path of the booth_data in cwd")
    parser.add_argument("--poll_data", type=str, default=None, help="Path of the opinion polling in cwd")
    parser.add_argument("--historical_results", type=str, default=None, help="Historical results data")

    # INANNA Arguments (Optional Flags)
    parser.add_argument("--flow_matrix", type=str, default=None, help="Vote preference flows (OPV)")
    parser.add_argument("--backtest_twocp", type=str, default=None, help="Seat 2CP final dictionary string")

    args = parser.parse_args()

    # --- BUILD ENKI ARGUMENTS DYNAMICALLY ---
    enki_kwargs = {}
    if args.seat_name is not None:
        enki_kwargs["seat_name"] = args.seat_name
    if args.output_root is not None:
        enki_kwargs["output_root"] = args.output_root
    if args.booth_data is not None:
        enki_kwargs["booth_data"] = args.booth_data
    if args.poll_data is not None:
        enki_kwargs["poll_data"] = args.poll_data
    if args.historical_results is not None:
        enki_kwargs["historical_results"] = args.historical_results

    # Fires ENKI using only what was typed; anything else drops back to ENKI's defaults
    ENKI(**enki_kwargs)


    # --- BUILD INANNA ARGUMENTS DYNAMICALLY ---
    inanna_kwargs = {}
    if args.seat_name is not None:
        inanna_kwargs["seat_name"] = args.seat_name
    if args.output_root is not None:
        inanna_kwargs["output_root"] = args.output_root
    if args.flow_matrix is not None:
        inanna_kwargs["flow_matrix"] = args.flow_matrix
        
    if args.backtest_twocp is not None:
        try:
            inanna_kwargs["backtest_twocp"] = ast.literal_eval(args.backtest_twocp)
        except (ValueError, SyntaxError):
            parser.error("The --backtest_twocp argument must be a valid dictionary string (e.g. '{\"2022\": 52.3}').")

    # Fires INANNA using only what was typed; anything else drops back to INANNA's defaults
    INANNA(**inanna_kwargs)

if __name__ == "__main__":
    main()
