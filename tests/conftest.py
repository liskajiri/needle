from hypothesis import settings

settings.register_profile("ci", max_examples=2, deadline=200)
settings.register_profile("default", max_examples=5, deadline=500)
settings.register_profile("dev", max_examples=100, deadline=None)

settings.load_profile("default")
