# LAB-02-Advanced-Python

Source notebook: LAB-02-Advanced-Python.ipynb

# Day 2 Lab — Advanced Python
## Iterators, Generators, Decorators & Context Managers — Hands-On Lab

This lab is a large, standalone set of exercises to reinforce Day 2's Advanced Python topics. Work through each part in order; solutions are provided below each task.

### Structure
- Part A: Iterators (Exercises 1–3)
- Part B: Generators (Exercises 4–7)
- Part C: Decorators (Exercises 8–12)
- Part D: Context Managers (Exercises 13–16)
- Part E: Mini Challenges — Combine Everything (Exercises 17–20)

## Part A — Iterators

### Exercise 1: `EvenNumbers` Iterator
Build a custom iterator class `EvenNumbers` that iterates over even numbers from 0 up to (and including) a given limit.

### Exercise 2: `CyclicIterator`
Build an iterator that cycles through a list forever (careful: infinite!). Then use `itertools.islice` (or a manual counter) to only take the first 7 values so it doesn't run forever.

### Exercise 3: Make a Custom Class Iterable (Without Being an Iterator Itself)
Build a `WeekDays` class whose `__iter__` method returns a **separate iterator object**, so multiple independent loops over the same `WeekDays` instance don't interfere with each other.

## Part B — Generators

### Exercise 4: Fibonacci Generator
Write a generator function `fibonacci(n)` that yields the first `n` Fibonacci numbers, without storing them all in a list first.

### Exercise 5: Infinite Generator With a Stop Condition
Write an infinite generator `natural_numbers()` that yields 1, 2, 3, ... forever. Then write code that consumes it but stops once it finds the first number greater than 1000 that is divisible by 7 and 11.

### Exercise 6: File-Line Style Generator (Simulated)
Simulate reading a huge log file line by line using a generator, then write a generator pipeline: one generator yields raw lines, a second generator filters only lines containing the word `"ERROR"`.

### Exercise 7: Generator Expression for Data Processing
Given a list of dictionaries representing sensor readings, use a generator expression (not a list comprehension) to compute the average temperature without building an intermediate list.

## Part C — Decorators

### Exercise 8: `uppercase_result` Decorator
Write a decorator `uppercase_result` that converts a function's string return value to uppercase.

### Exercise 9: `retry` Decorator
Write a decorator `retry(max_attempts)` (a decorator factory) that retries the decorated function up to `max_attempts` times if it raises an exception, printing which attempt failed, and finally re-raising if all attempts fail.

### Exercise 10: Authentication / Access Control Decorator
Simulate a simple permission system: write a decorator `require_role(role)` that only allows a function to run if a global `current_user` dict has a matching `"role"` key; otherwise it prints an "Access Denied" message and does not call the function.

### Exercise 11: Caching Decorator (Manual, Without `functools`)
Write your own caching decorator `simple_cache` from scratch (without using `functools.lru_cache`) that stores results in a dictionary keyed by the function's arguments.

### Exercise 12: Logging Decorator That Preserves Metadata
Write a decorator `logged` that prints the function name, arguments, and return value every time it's called — and correctly preserves the original function's `__name__` and docstring using `functools.wraps`.

## Part D — Context Managers

### Exercise 13: `Timer` Context Manager With Exception Safety
Write a class-based context manager `Timer` that prints elapsed time on exit **even if an exception occurs** inside the `with` block (but doesn't suppress the exception).

### Exercise 14: `open_file_safe` Context Manager
Write a context manager class `open_file_safe` that mimics `open()` — it should open a file on `__enter__` and guarantee it's closed on `__exit__`, printing a confirmation message when the file is closed.

### Exercise 15: `contextlib`-Based Database Connection Simulator
Using `@contextmanager` from `contextlib`, write a `db_connection(name)` context manager that prints "Connecting to <name>" on entry and "Closing connection to <name>" on exit, even if an error happens inside.

### Exercise 16: Nested Context Managers
Use two context managers together in a single `with` statement (e.g., two simulated resources) and observe the order in which `__enter__` and `__exit__` are called.

## Part E — Mini Challenges (Combine Everything)

### Exercise 17: Lazy Data Pipeline
Build a data-processing pipeline using **only generators**:
1. `read_numbers()` — yields numbers 1 to 50
2. `filter_multiples_of_3(numbers)` — yields only multiples of 3
3. `square(numbers)` — yields each number squared

Chain them together and print the final results.

### Exercise 18: Timing + Retry Decorator Stack
Combine the `timer` and `retry` decorators from today so a function is both timed AND retried on failure. Apply both decorators to a function that randomly fails.

### Exercise 19: Context Manager + Generator Together — Batch File Writer
Build a context manager `BatchWriter(path)` that, on `__enter__`, opens a file and returns a generator-friendly `write_batch(lines)` method that writes multiple lines at once. On `__exit__`, it should close the file and print how many total lines were written.

### Exercise 20: Final Challenge — Rate-Limited API Call Simulator
Build a complete mini-system combining everything from today:
- A generator `simulate_api_requests(n)` that yields request IDs 1..n
- A decorator `rate_limited(seconds)` that ensures the decorated function waits at least `seconds` between calls (use `time.time()` to track the last call time)
- A context manager `APISession()` that prints "Session started" on enter and "Session closed, X requests processed" on exit (track count via an attribute)

Use all three together to process the simulated requests.

---
## Lab Wrap-Up

You've now practiced:
- Writing custom iterators and making classes iterable
- Building lazy, memory-efficient generators, including pipelines and infinite sequences
- Writing decorators for logging, caching, retries, access control, and rate limiting — including decorator factories and stacking multiple decorators
- Managing resources safely with class-based and `contextlib`-based context managers
- Combining generators, decorators, and context managers together in realistic mini-systems

These patterns show up constantly in production Python code — including AI/LLM pipelines (batching data with generators, retrying API calls with decorators, and managing resources like model sessions or connections with context managers).
