# 📋 Change Log — Bhavesh Dev Branch
> Branch: `bhavesh-dev`  
> Purpose: Fix pipeline errors so the full generation flow works end-to-end  
> Rule: All changes made locally first → pushed to `bhavesh-dev` → merged to `main` when stable

---

## ✅ Changes Already Done (on `main` branch, done earlier today)

### 1. `.env.cpu` — Fixed Redis Loopback Timeout
- **File:** `.env.cpu`
- **Problem:** CPU machine was using its own Public IP (`18.207.13.85`) to connect to Redis. AWS blocks a server from talking to itself via Public IP — this caused `Timeout` errors on the website.
- **Fix:** Changed all Redis URLs from `redis://18.207.13.85:6380/0` → `redis://localhost:6380/0`
- **Result:** Website stopped showing Timeout errors. Tasks could be queued.

### 2. `.env.gpu` — Fixed Wrong Redis Host & Port
- **File:** `.env.gpu`
- **Problem:** GPU worker config was pointing to `127.0.0.1:6379` (its own machine, wrong port). Redis is on the CPU machine at port `6380`.
- **Fix:** Updated all Redis URLs to `redis://18.207.13.85:6380/0`
- **Result:** GPU worker can now reach the CPU machine's Redis.

### 3. `.env.gpu` — Added Missing `REDIS_HOST` and `REDIS_PORT` Variables
- **File:** `.env.gpu`
- **Problem:** The GPU worker script reads `REDIS_HOST` and `REDIS_PORT` as **separate variables**, not just the URL. These were missing from the file. Worker silently fell back to hardcoded default `localhost:6379`.
- **Fix:** Added `REDIS_HOST=18.207.13.85` and `REDIS_PORT=6380` explicitly.
- **Result:** Worker now reads correct host and port from config.

---

## 🔧 Changes In Progress (on `bhavesh-dev` branch)

### 4. `worker/run_manual_worker.py` — Force Reload of Environment Variables
- **File:** `worker/run_manual_worker.py`
- **Problem:** `load_dotenv()` does not override environment variables already set in the OS. On restart, worker kept using stale cached value `localhost:6379` instead of reading fresh `6380` from `.env`.
- **Fix:** Changed `load_dotenv(_env_path)` → `load_dotenv(_env_path, override=True)`
- **Status:** ✅ Done on `bhavesh-dev`

### 5. `worker/workers/base_worker.py` — Force Reload of Environment Variables
- **File:** `worker/workers/base_worker.py`
- **Problem:** Same issue as above. `base_worker.py` is the file that **actually connects to Redis** using `REDIS_HOST` and `REDIS_PORT`. It had its own `load_dotenv()` call without `override=True`, so it was reading stale port 6379.
- **Fix:** Changed `load_dotenv()` → `load_dotenv(override=True)`
- **Status:** ✅ Done on `bhavesh-dev`

---

## 📝 Known Issues Still Open

| Issue | Description | Status |
|---|---|---|
| Task stays "queued" | GPU worker not picking up tasks from Redis | ✅ Fixed in Change #6 & #7 |
| Service reads `.env` not `.env.gpu` | Need `cp .env.gpu .env` after every pull | 📋 Future automation task |

---

## 🏗️ Architecture Notes (For Reference)
```
Website (Gradio UI on CPU)
  ↓  clicks "Queue SD Stage 1"
FastAPI (CPU) → pushes task to Redis on localhost:6379
  ↓
GPU Worker picks task from Redis at 172.31.92.14:6379 (private IP, always reachable)
  ↓  runs AI model
Uploads image to S3
  ↓
Updates MongoDB with image URL + status = "done"
  ↓
Website reads MongoDB → shows image
```

---

*Last updated: 2026-05-05*
