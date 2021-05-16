# multi-graph-former

**USE AT YOUR OWN RISK** Intra and inter graph attention, vert updates, and edge updates with dynamic structure. 

The idea is: recurrent neural networks excel at remembering several bits of information in their recurrent attractor space. However they fall short o memorizing information that can be compositionally expressed and modified based on input and existing recurrent information. My hypothesis is that using a graph-structured (instead of dense) recurrent variable would enable RNN's to excel.

## Getting Started

1. Download this repository
2. Open terminal inside the root directory `multi-graph-former`
3. Run one of the tests
  - `$ ~/Downloads/multi-graph-former python3 modules/language_wm_graph_former_test.py`
  - `$ ~/Downloads/multi-graph-former python3 modules/language_wm_graph_former_test2.py`

Alternatively, copy the `layers` and `utils` directories into your own project.
