"""Highway-env observation de-normalisation and episode limits."""

# highway-env normalises vx into [-1, 1] using [-2*MAX_SPEED, 2*MAX_SPEED] and
# x into [-1, 1] using [-5*MAX_SPEED, 5*MAX_SPEED], with Vehicle.MAX_SPEED = 40 m/s.
HIGHWAY_SPEED_SCALE = 80.0
HIGHWAY_DIST_SCALE = 200.0
HIGHWAY_MAX_STEPS = 300
