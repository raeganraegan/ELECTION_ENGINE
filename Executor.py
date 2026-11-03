"""Executor.py -- run ENKI for one seat. Edit the dict below, then run this file."""
from enki import ENKI

def main():
    balmain = {
        "seat_name": "Balmain",
        "booth_results": "balmain.csv",
        "polls": "sample_polls.csv",
        "parties": ("ALP", "LNP", "GRN", "OTHER"),     # the LAST one is the reference category

        # One Nation: "none" | "brian" (fixed vote transfer) | "entry" (needs an ONP column in the polls)
        "onp_mode": "none",
        "onp_state_denominator": False,

        "output_root": "forecast_output",
        "registered_electors": 59200
    }
    
    easthills = {
            "seat_name": "sample",
            "booth_results": "sample_booth_results.csv",
            "polls": "sample_polls.csv",
            "parties": ("ALP", "LNP", "GRN", "OTHER"),     # the LAST one is the reference category
    
            # One Nation: "none" | "brian" (fixed vote transfer) | "entry" (needs an ONP column in the polls)
            "onp_mode": "entry",
            "onp_state_denominator": True,
    
            "output_root": "forecast_output",
            "registered_electors": 60500
        }
    return ENKI(easthills)


if __name__ == "__main__":
    main()
