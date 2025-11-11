# 🚀 Deployment Guide for Render

## Prerequisites

- GitHub account
- Render account (free tier works fine)
- Modal inference service already deployed (✅ already done)

## Quick Deploy Steps

### Option 1: Using render.yaml (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add Render deployment config"
   git push origin main
   ```

2. **Deploy on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Select `model_backend` repository
   - Render will auto-detect `render.yaml` and configure everything
   - Click "Apply" to deploy

### Option 2: Manual Setup

1. **Create New Web Service**:
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select `model_backend`

2. **Configure Service**:
   ```
   Name: gigsama-backend
   Region: Oregon (or closest to you)
   Branch: main
   Root Directory: (leave empty if files are in repo root)
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn dual_backend:app --host 0.0.0.0 --port $PORT
   ```

   **IMPORTANT**: If your files are inside a `model_backend` folder in your repo:
   - Set Root Directory to: `model_backend`
   
   If your files are in the repo root (dual_backend.py is in the root):
   - Leave Root Directory **empty**

3. **Set Environment Variables**:
   - Click "Environment" tab
   - Add:
     ```
     MODAL_INFERENCE_URL = https://affum3331--gigsama-backend-fastapi-app.modal.run
     ```

4. **Deploy**:
   - Click "Create Web Service"
   - Wait 2-3 minutes for build and deployment

## Verify Deployment

Once deployed, your backend will be available at:
```
https://gigsama-backend.onrender.com
```

Test it:
```bash
curl https://gigsama-backend.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-11T...",
  "modal_url": "https://affum3331--gigsama-backend-fastapi-app.modal.run",
  "backends": {...}
}
```

## Test End-to-End

```bash
curl -X POST https://gigsama-backend.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "system": "You are a recruitment assistant. Extract job parameters as JSON.",
    "messages": [
      {"role": "user", "content": "Find backend developers in Kenya"}
    ]
  }'
```

## Architecture After Deployment

```
Frontend/Client
    ↓
Render Backend (gigsama-backend.onrender.com)
    ↓
Modal Inference (affum3331--gigsama-backend-fastapi-app.modal.run)
    ↓
Qwen3 1.7B on T4 GPU
```

## Free Tier Limits

Render Free Tier includes:
- ✅ 750 hours/month (enough for 24/7 operation)
- ✅ Automatic HTTPS
- ✅ Auto-deploy from Git
- ⚠️ Spins down after 15 min of inactivity (cold start ~30 seconds)
- ⚠️ 512MB RAM (sufficient for this lightweight backend)

## Troubleshooting

### Deployment Fails

Check build logs in Render dashboard. Common issues:
- Wrong Python version → Add `PYTHON_VERSION=3.11.0` env var
- Missing dependencies → Verify `requirements.txt`

### Backend Returns 500 Errors

- Check Render logs for errors
- Verify `MODAL_INFERENCE_URL` environment variable is set correctly
- Test Modal endpoint directly: `curl https://affum3331--gigsama-backend-fastapi-app.modal.run/health`

### Slow Response Times

- First request after idle: ~30-40 seconds (cold start on both Render and Modal)
- Subsequent requests: ~5-10 seconds
- Keep backend warm: Set up uptime monitoring (e.g., UptimeRobot pinging `/health` every 5 min)

## Cost Optimization

For production with no cold starts:
- Upgrade Render to Starter plan ($7/mo) for always-on service
- Modal inference stays free for moderate usage (<100 requests/day)

## Files Created

- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Process configuration
- ✅ `render.yaml` - Render deployment blueprint
- ✅ `DEPLOYMENT.md` - This guide

## Next Steps

1. Deploy to Render using steps above
2. Update your frontend to use: `https://gigsama-backend.onrender.com/chat`
3. Test thoroughly
4. Set up monitoring (optional)

---

**Status**: Ready to deploy! 🚀
