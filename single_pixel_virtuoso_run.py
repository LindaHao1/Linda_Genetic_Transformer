from virtuoso_engine import VirtuosoPixelEngine

# engine = VirtuosoPixelEngine("virtuoso_engine/config.yaml")

# result = engine.run()                                    # defaults
# result = engine.run({"nmos": {"width": "6u"}})           # inline override
# print(result)
# # results = engine.sweep(["nmos", "width"], ["3u","6u","9u"])  # parameter sweep

with VirtuosoPixelEngine("virtuoso_engine/config.yaml") as engine:
    # results = engine.sweep(["nmos", "width"], ["3u", "6u", "9u"])
    # results = engine.sweep_configs([
    # {"nmos": {"width": "3u", "fingers": 4}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 5}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 6}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 7}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 8}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 9}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 10}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 11}, "mim_cap": {"length": "15u", "width": "15u"} },
    # {"nmos": {"width": "3u", "fingers": 12}, "mim_cap": {"length": "15u", "width": "15u"} }
    # ])

    engine.run()
    # Virtuoso starts once, all 3 × 2 scripts fed through it
# engine.close() called automaticall