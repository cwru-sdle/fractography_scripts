This repo contains the scripts used to analyze fatigue fracture surfaces

Organize_data needs to be restructured to make the data tidy. I can still merge everything on Sample#, but need to rename that column to sample_num. Having special characters is causing issues. The columns should be:

1. sample_id ('Sample#'): string, factor
2. build_id ('Build ID'): string, factor
3. build_plate_position ('Build #'): string, factor
4. testing_position ('Test #'): int, factor
4. scan_power_W ('Scan Power (W)'): float
5. scan_velocity_mm_s ('Scan velocity (mm/s)'): float
6. energy_density_J_mm3 (scan_power/scan_velocity * hatch_spacing * layer thickess): float
7. test_stress_Mpa ('mpa','σ max initiation (MPa)'): float
8. cycles (Cycles, Cycles @ Failure): int
9. image_class: string, factor
10. image_path: string
11. image_basename: string
12. points: string ->np.array