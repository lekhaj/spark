import subprocess

steps = [
    ("LLM Generation", "python llm_generate_single.py"),
    ("JSON Finalization", "python json_finalizer.py"),
    ("Send to MongoDB", "python send_to_db.py")
]

for label, command in steps:
    print(f"\\n🚀 Step: {label}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed at step: {label}")
        break
    else:
        print(f"✅ Completed: {label}")