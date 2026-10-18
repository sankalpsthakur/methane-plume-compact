"""Tests the soft-deadline function of kernels/train_gpu/main.py with real toy processes: one that dies on SIGTERM, one that ignores SIGTERM (needs SIGKILL), one that ignores everything it can.
The function source is extracted from the driver with ast, so what is tested is the code that ships."""
import ast, os, subprocess, sys, time
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "train_gpu", "main.py")).read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "enforce_deadline")
ns = {"time": time}; exec(compile(ast.Module([fn], []), "driver", "exec"), ns); enforce = ns["enforce_deadline"]
def proc(script): return subprocess.Popen(["bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def item(name, p): return (name, p, 0.0, None, 0, "cpu")
def wait_dead(p, s=3.0):
    t = time.time()
    while p.poll() is None and time.time() - t < s: time.sleep(0.05)
    return p.poll() is not None
logs = []; log = logs.append
# 1. before the deadline nothing happens
p1 = proc("sleep 30"); running = [item("a", p1)]; pending = [{"name": "p1"}]; status = {}; st = {}
assert enforce(running, pending, status, 5.0, 0.0, 100.0, st, log=log) == pending and p1.poll() is None and not status
# 2. deadline: SIGTERM, pending marked not started and emptied
out = enforce(running, pending, status, 101.0, 0.0, 100.0, st, log=log)
assert out == [] and status["p1"]["note"].startswith("not started") and wait_dead(p1) and p1.returncode == -15 and st["hit"] == 101.0
# 3. a process that ignores SIGTERM survives it, is SIGKILLed after term_grace, and is abandoned (not waited for) after give_up
p2 = proc("trap '' TERM; while true; do sleep 1; done"); time.sleep(0.5); running = [item("stubborn", p2)]; status = {}; st = {}
enforce(running, [], status, 200.0, 0.0, 100.0, st, term_grace=2, give_up=4, log=log)           # hit: SIGTERM, ignored
time.sleep(0.3); assert p2.poll() is None
enforce(running, [], status, 201.5, 0.0, 100.0, st, term_grace=2, give_up=4, log=log); assert p2.poll() is None   # inside the grace
enforce(running, [], status, 203.0, 0.0, 100.0, st, term_grace=2, give_up=4, log=log); assert wait_dead(p2) and p2.returncode == -9   # past the grace: SIGKILL
# 4. a process that cannot be killed (simulated by a stub whose kill does nothing) is abandoned after give_up, so the main loop can end
class Stuck:
    def terminate(self): pass
    def kill(self): pass
    def poll(self): return None
running = [item("stuck", Stuck())]; status = {}; st = {}
enforce(running, [], status, 300.0, 0.0, 100.0, st, term_grace=2, give_up=4, log=log)
enforce(running, [], status, 305.0, 0.0, 100.0, st, term_grace=2, give_up=4, log=log)
assert running == [] and status["stuck"]["note"].startswith("did not exit after SIGKILL") and any("gave up" in m for m in logs)
# 5. nothing queued or running -> no-op
assert enforce([], [], {}, 999.0, 0.0, 100.0, {}, log=log) == [] and enforce([], [{"name": "x"}], {}, 999.0, 0.0, None, {}, log=log) == [{"name": "x"}]
for p in (p1, p2):
    if p.poll() is None: p.kill()
print("driver deadline: ok")
