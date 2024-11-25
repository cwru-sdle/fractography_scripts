This repo contains the scripts used to analyze fatigue fracture surfaces

Organize_data needs to be restructured to make the data tidy. I can still merge everything on Sample#, but need to rename that column to sample_num. Having special characters is causing issues. The columns should be:

1. sample_id ('Sample#'): string, factor
2. build_id ('Build ID'): string, factor
3. sample_position ('Build #'): string, factor
4. test_num ('Test #', 'Retest'): int
5. hatch_spacing ('Hatch Spacing (mm)'): float
6. scan_power ('Scan Power (W)'): float
7. scan_velocity ('Scan velocity (mm/s)'): float
8. energy_density (scan_power/scan_velocity * hatch_spacing * layer thickess): float
9. test_stress ('mpa','σ max initiation (MPa)'): float
10. R_ratio: float
11. cycles_failure (Cycles, Cycles @ Failure): int
12. test_result ('Failed?'): boolean
13. image_class: string, factor
14. image_path: string
15. image_basename: string.