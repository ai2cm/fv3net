from inspect import signature

import generate_movie_stills


def test__movie_funcs():
    movie_funcs = generate_movie_stills._movie_funcs()
    for name, func in movie_funcs.items():
        sig = signature(func)
        assert len(sig.parameters) == 3
