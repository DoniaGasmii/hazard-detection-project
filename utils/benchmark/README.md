# Interactive Benchmark Dashboard (Streamlit)

## Run
```bash
pip install -r utils/benchmark/interactive/requirements.txt
streamlit run utils/benchmark/interactive/streamlit_benchmark_app.py
```
## Use

1. Upload one or more **results CSVs** (must include `model` + numeric metrics).  
2. (Optional) Upload **errors CSV** with `model,true,pred[,confidence]` for confusion matrix.  
3. Choose **Group by**, **Color by**, **Epoch column**, and **Metrics** in the sidebar.  
4. Explore interactive **bar/line/box** plots, summary tables, and errors view.  
