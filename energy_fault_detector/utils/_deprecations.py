
import functools
import inspect
import warnings
from typing import Mapping


def deprecate_kwargs(mapping: Mapping[str, str], *, prefer: str = "old"):
    """Decorator to deprecate parameter names. Works for functions and methods (keeps 'self' in bound arguments).

    Args:
        mapping: {old_name: new_name}
        prefer: "old" or "new" to resolve conflicts when both provided.
    """

    if prefer not in {"old", "new"}:
        raise ValueError("prefer must be 'old' or 'new'")

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            provided = dict(bound.arguments)  # includes 'self' for methods if passed

            # start from what was provided; we'll call the function with these resolved arguments
            call_kwargs = dict(provided)

            for old, new in mapping.items():
                if old not in provided:
                    continue

                old_val = provided[old]
                new_passed = new in provided
                new_val = provided.get(new)

                # usage warning
                warnings.warn(f"'{old}' is deprecated; use '{new}' instead",
                              DeprecationWarning, stacklevel=3)

                if not new_passed:
                    call_kwargs[new] = old_val
                    call_kwargs.pop(old, None)
                else:
                    if old_val != new_val:
                        if prefer == "old":
                            warnings.warn(
                                f"Both '{old}' (deprecated) and '{new}' were provided with different values; "
                                f"the deprecated '{old}' value will be used.",
                                DeprecationWarning, stacklevel=3,
                            )
                            call_kwargs[new] = old_val
                            call_kwargs.pop(old, None)
                        else:
                            warnings.warn(
                                f"Both '{old}' (deprecated) and '{new}' were provided with different values; "
                                f"the explicit '{new}' value will be used and '{old}' ignored.",
                                DeprecationWarning, stacklevel=3,
                            )
                            call_kwargs.pop(old, None)
                    else:
                        # identical: prefer new name, drop old
                        call_kwargs.pop(old, None)
                        call_kwargs[new] = new_val

            # Call original with resolved bound arguments (positional converted to keywords)
            return func(**call_kwargs)

        return wrapper

    return decorator
