from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter

import pdb

rossi_dataset = load_rossi()
cph = CoxPHFitter()
cph.fit(rossi_dataset, duration_col='week', event_col='arrest')

pdb.set_trace()