# 📥 Raw Data Source

The raw dataset used in this project is the Expedia Flight Prices dataset, downloaded directly as an Apache Parquet file (Snappy compressed):

- **Dataset Link**: [Expedia Flight Prices Dataset](https://www.dropbox.com/scl/fo/mybc5v9s800orsu78b6ao/h?rlkey=1an4ndcscd5uw9yi7oxx8ypfn&e=1&dl=0)
- **Format**: Apache Parquet (Snappy compressed)
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Collection Date**: April–October 2022
- **Notes**:  
  - Dataset includes airfare records for flights between major U.S. airports.
  - Focused specifically on flights departing from LAX.

**Loading the dataset directly:**
```python
import pandas as pd
df = pd.read_parquet("path/to/data.parquet", engine="pyarrow")

