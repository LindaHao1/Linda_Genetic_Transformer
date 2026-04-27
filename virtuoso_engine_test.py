from virtuoso_engine import VirtuosoPixelEngine

# engine = VirtuosoPixelEngine("virtuoso_engine/config.yaml")

# result = engine.run()                                    # defaults
# result = engine.run({"nmos": {"width": "6u"}})           # inline override
# print(result)
# # results = engine.sweep(["nmos", "width"], ["3u","6u","9u"])  # parameter sweep

# with VirtuosoPixelEngine("virtuoso_engine/config.yaml") as engine:
#     # results = engine.sweep(["nmos", "width"], ["3u", "6u", "9u"])
#     # results = engine.sweep_configs([
#     # {"nmos": {"width": "3u", "fingers": 4}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 5}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 6}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 7}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 8}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 9}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 10}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 11}, "mim_cap": {"length": "15u", "width": "15u"} },
#     # {"nmos": {"width": "3u", "fingers": 12}, "mim_cap": {"length": "15u", "width": "15u"} }
#     # ])

#     engine.run()
#     # Virtuoso starts once, all 3 × 2 scripts fed through it
# # engine.close() called automaticall


from virtuoso_engine import ParallelVirtuosoEngine

# with ParallelVirtuosoEngine("virtuoso_engine/config.yaml") as engine:
#     results = engine.run_batch([
#         {"pixelated": {"state_list": [[0, 0], [1, 0]]}},
#         # {"pixelated": {"state_list": [[1, 0]]}},
#         {"pixelated": {"state_list": [[0, 1], [1, 1]]}},
#         # {"pixelated": {"state_list": [[1, 1]]}},
#     ])

# --- Cap pixel pairs test ---
# Adds one cap (1 pF) between pixel (0,1) and pixel (1,2) in the 3×5 default matrix.
# Both pixels are type-1 (metal), so they already have patches — no matrix change needed.
# Expected: EMX sees 10 ports (8 base + CP_0A + CP_0B); each state's .scs contains
#           "c_pair0 (cp_0a cp_0b) capacitor c=1e-12"
# with VirtuosoPixelEngine("virtuoso_engine/config.yaml") as engine:
#     # result = engine.run({
#     #     "pixelated": {
#     #         "cap_pixel_pairs": [[[0, 1], [1, 2]]],
#     #         "cap_pair_values": [1.0e-12],
#     #     }
#     # })
#     result = engine.run({})
#     print("Cap pair run success:", result.success)
#     print("S-param files:", result.s_param_files)


### FULL DESIGN TEST ###
from virtuoso_engine.engine import VirtuosoPixelEngine

with VirtuosoPixelEngine("virtuoso_engine/config_base.yaml") as engine:
    # result = engine.run("virtuoso_optimization_GA/configs/test/full_layout_test.yaml")
    result = engine.run("/disk/mcao53/Sim_engine_results/test_sims/cand2/sim_20260425_222018_config.yaml")
    print(result.success, result.run_dir)