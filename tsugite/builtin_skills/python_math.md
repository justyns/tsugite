---
name: python_math
description: Reference for performing common math operations in Python
---

# Python Math Operations

Use these patterns to perform accurate math in Python:

## Basic Arithmetic

```python
x, y = 10, 3
sum_ = x + y        # 13
diff = x - y        # 7
product = x * y     # 30
quotient = x / y    # 3.3333333333333335 (float)
floor_div = x // y  # 3 (integer division)
remainder = x % y   # 1
power = x ** y      # 1000
```

## Math Module Essentials

```python
import math

sqrt_val = math.sqrt(16)      # 4.0
log_val = math.log(100, 10)   # 2.0
angle = math.radians(45)      # Convert to radians
sin_val = math.sin(angle)     # 0.707...
constants = (math.pi, math.e) # Common constants
```

Key functions worth memorizing:

- `math.ceil`, `math.floor`, `math.trunc` for rounding and chopping decimals
- `math.fsum` for accurate summations, `math.prod` for products across iterables
- `math.pow`, `math.exp`, `math.log`, `math.log2`, `math.log10` for exponent and log work
- `math.isclose` (tolerance-aware comparisons), `math.isfinite`, `math.isnan` for validation
- `math.gcd`, `math.comb`, `math.perm` for number theory and counting problems
- `math.dist`, `math.hypot` for Euclidean distance and vector norms
- Constants: `math.pi`, `math.tau`, `math.e`, plus `math.inf` and `math.nan`

## Precise Decimal Arithmetic

```python
from decimal import Decimal, getcontext

getcontext().prec = 28  # Set precision
price = Decimal("19.99")
qty = Decimal("3")
total = price * qty  # 59.97 without floating error
```

## Fractions and Rational Numbers

```python
from fractions import Fraction

ratio = Fraction(3, 4) + Fraction(5, 6)
ratio = ratio.limit_denominator()  # Simplify result
```

## Aggregations

```python
import math

numbers = [2, 4, 6, 8]

count = len(numbers)
total = sum(numbers)
mean = total / count
precise_total = math.fsum(numbers)  # Lossless summation when floats are involved
```

Prefer `math.fsum` for floating-point sums and `statistics.fmean` when averaging precise floats.

## Complex Numbers

```python
import cmath

z = 3 + 4j  # Complex number literal
z = complex(3, 4)  # Or construct explicitly

magnitude = abs(z)  # 5.0
phase = cmath.phase(z)  # 0.927... (in radians)
real_part = z.real  # 3.0
imag_part = z.imag  # 4.0

# Complex math functions
sqrt_neg = cmath.sqrt(-1)  # 1j
exp_complex = cmath.exp(1j * cmath.pi)  # -1+1.2e-16j (Euler's formula)
```

## Statistics Module

```python
from statistics import mean, median, mode, stdev, variance, quantiles

data = [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9]

avg = mean(data)  # 4.75
mid = median(data)  # 5.0
most_common = mode(data)  # 5
std_dev = stdev(data)  # 2.56...
var = variance(data)  # 6.57...

# Quantiles (quartiles, percentiles)
quartiles = quantiles(data, n=4)  # [2.75, 5.0, 6.5]
percentiles = quantiles(data, n=100)  # 99 percentile values

# Floating-point mean (more accurate for float data)
from statistics import fmean
float_data = [1.1, 2.2, 3.3]
precise_mean = fmean(float_data)  # More accurate than sum()/len()
```

Common use cases:
- `mean()` - Average value
- `median()` - Middle value (robust to outliers)
- `mode()` - Most frequent value
- `stdev()` / `variance()` - Measure of spread
- `quantiles()` - Percentiles, quartiles, deciles

## Random Number Generation

```python
import random

# Basic random operations
random.random()  # Float in [0.0, 1.0)
random.uniform(1.5, 10.5)  # Float in [1.5, 10.5]
random.randint(1, 10)  # Integer in [1, 10] (inclusive)
random.randrange(0, 100, 5)  # Random from range(0, 100, 5)

# Sequences
items = ['a', 'b', 'c', 'd']
random.choice(items)  # Pick one random element
random.choices(items, k=3)  # Pick 3 with replacement
random.sample(items, k=2)  # Pick 2 without replacement
random.shuffle(items)  # In-place shuffle

# Distributions
random.gauss(0, 1)  # Normal distribution (mean=0, stdev=1)
random.expovariate(1.5)  # Exponential distribution
random.triangular(1, 10, 5)  # Triangular distribution (low, high, mode)

# Seeding for reproducibility
random.seed(42)  # Set seed for deterministic results
```

### Cryptographically Secure Random

```python
import secrets

# Use secrets for security-sensitive operations (tokens, passwords, keys)
token = secrets.token_hex(16)  # 32-character hex string
token_bytes = secrets.token_bytes(16)  # 16 random bytes
token_url = secrets.token_urlsafe(16)  # URL-safe token

# Random choice (cryptographically secure)
password_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
password = ''.join(secrets.choice(password_chars) for _ in range(12))

# Random numbers
secure_int = secrets.randbelow(100)  # Random int in [0, 100)
secure_choice = secrets.choice(['option1', 'option2', 'option3'])
```

**Rule of thumb:** Use `secrets` for security (passwords, tokens), `random` for simulations/games.

## Advanced Math Libraries

- `numpy`: vectorized arithmetic, linear algebra, random sampling, array operations
- `scipy`: optimization, integration, statistics, signal processing, interpolation
- `statistics`: quick descriptive stats (mean, median, stdev) - part of standard library

Prefer these when problems move from scalar math into data analysis or scientific workflows.

### NumPy Quick Examples

```python
import numpy as np

# Arrays and vectorized operations
arr = np.array([1, 2, 3, 4, 5])
squared = arr ** 2  # [1, 4, 9, 16, 25]
total = arr.sum()  # 15
average = arr.mean()  # 3.0

# Linear algebra
matrix = np.array([[1, 2], [3, 4]])
inverse = np.linalg.inv(matrix)
determinant = np.linalg.det(matrix)

# Random sampling
samples = np.random.normal(0, 1, size=1000)  # 1000 samples from N(0,1)
```