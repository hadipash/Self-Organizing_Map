# representation of letters as a grid
inp = [
    # Font 1
    [
        # A
        [0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 0, 1, 1, 1],
        # B
        [1, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 0],
        # C
        [0, 0, 1, 1, 1, 1, 1,
         0, 1, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 1,
         0, 0, 1, 1, 1, 1, 1],
        # D
        [1, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 1, 1, 0, 0],
        # E
        [1, 1, 1, 1, 1, 1, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 1, 1, 0, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 1],
        # J
        [0, 0, 0, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 0, 0],
        # K
        [1, 1, 1, 0, 0, 1, 1,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 1, 0, 0, 0, 0,
         0, 1, 1, 0, 0, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 0, 0, 1, 1]
    ],
    # Font 2
    [
        # A
        [0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0],
        # B
        [1, 1, 1, 1, 1, 1, 0,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 0,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 0],
        # C
        [0, 0, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 0, 0],
        # D
        [1, 1, 1, 1, 1, 0, 0,
         1, 0, 0, 0, 0, 1, 0,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 1, 0,
         1, 1, 1, 1, 1, 0, 0],
        # E
        [1, 1, 1, 1, 1, 1, 1,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1],
        # J
        [0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 0, 0],
        # K
        [1, 0, 0, 0, 0, 1, 0,
         1, 0, 0, 0, 1, 0, 0,
         1, 0, 0, 1, 0, 0, 0,
         1, 0, 1, 0, 0, 0, 0,
         1, 1, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 0, 0, 0,
         1, 0, 0, 1, 0, 0, 0,
         1, 0, 0, 0, 1, 0, 0,
         1, 0, 0, 0, 0, 1, 0]
    ],
    # Font 3
    [
        # A
        [0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 1, 0, 0, 0, 1, 1],
        # B
        [1, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 0],
        # C
        [0, 0, 1, 1, 1, 0, 1,
         0, 1, 0, 0, 0, 1, 1,
         1, 0, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 0, 0],
        # D
        [1, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 1, 1, 0, 0],
        # E
        [1, 1, 1, 1, 1, 1, 1,
         0, 1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 1,
         1, 1, 1, 1, 1, 1, 1],
        # J
        [0, 0, 0, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 0, 0],
        # K
        [1, 1, 1, 0, 0, 1, 1,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 1, 0, 0, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 0, 0, 1, 1]
    ]
]
