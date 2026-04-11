# Quick Start Guide - Level Set 3D PINN

## 🚀 Recommended Workflow

### Step 1: Generate Data

```bash
python generate_levelset_data.py
```

### Step 2: Start Training (v5.5)

```bash
python train_levelset_3d.py --config config/v5.5_full.json
```

**Key Monitoring Metrics** (First 1000 epochs):
- ✅ Sign loss < 0.001
- ✅ Ink<0 > 95%
- ✅ Polar>0 > 95%

### Step 3: Monitor Training
 
 Check the logs directly:
 ```bash
 tail -f training_v5.5.log
 ```
 
 ## 📊 Expected Results
 
 | Metric | Target |
 |--------|--------|
 | **Sign Loss** | < 0.001 |
 | **Ink ψ < 0** | > 95% |
 | **0V Aperture** | < 10% |
 | **30V Aperture** | 70-90% |
 
 ## 📁 Useful Scripts
 
 | Script | Purpose |
 |--------|---------|
 | `evaluate.py` | Evaluate model performance |
 
 ## 📚 More Documentation

- [Project Review](docs/PROJECT_REVIEW_20260127.md)
- [Changelog](docs/CHANGELOG.md)
- [User Guide](docs/USER_GUIDE.md)

---

**Ready to start? Run the data generation command first!** 🎉
