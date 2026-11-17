# Hardware recording inventory

| Run (UTC directory timestamp) | Status | Drone | Real objects | Policy samples | Mocap samples |
| --- | --- | --- | --- | ---: | ---: |
| [deploy_20261117T100017.801841Z](../deploy_20261117T100017.801841Z/metadata.json) | finished | [2] | True | 256 | 532 |
| [deploy_20261117T100651.417694Z](../deploy_20261117T100651.417694Z/metadata.json) | no_flight | [2] | True | 0 | 0 |
| [deploy_20261117T100827.697140Z](../deploy_20261117T100827.697140Z/metadata.json) | finished | [2] | True | 265 | 533 |
| [deploy_20261117T102228.660876Z](../deploy_20261117T102228.660876Z/metadata.json) | ended_without_finish | [2] | False | 238 | 185 |
| [deploy_20261117T102459.085589Z](../deploy_20261117T102459.085589Z/metadata.json) | ended_without_finish | [2] | False | 2212 | 1374 |
| [deploy_20261117T104622.175276Z](../deploy_20261117T104622.175276Z/metadata.json) | finished | [3] | False | 244 | 209 |
| [deploy_20261117T104828.785539Z](../deploy_20261117T104828.785539Z/metadata.json) | finished | [3] | True | 246 | 206 |
| [deploy_20261117T110059.075060Z](../deploy_20261117T110059.075060Z/metadata.json) | finished | [3] | True | 297 | 237 |
| [deploy_20261117T110653.614444Z](../deploy_20261117T110653.614444Z/metadata.json) | finished | [3] | False | 235 | 185 |
| [deploy_20261117T111147.345218Z](../deploy_20261117T111147.345218Z/metadata.json) | finished | [3] | False | 224 | 178 |
| [deploy_20261117T113735.863815Z](../deploy_20261117T113735.863815Z/metadata.json) | ended_without_finish | [3] | True | 248 | 191 |
| [deploy_20261117T115247.042900Z](../deploy_20261117T115247.042900Z/metadata.json) | finished | [3] | True | 303 | 557 |
| [deploy_20261117T115447.164994Z](../deploy_20261117T115447.164994Z/metadata.json) | no_flight | [3] | True | 0 | 0 |

All 13 finalized raw directories are preserved, including the startup failure and interrupted flights.
Detailed reports cover selected runs; inclusion in this inventory does not imply a full causal analysis.

`complete` in a recorder summary means the recorder finalized without a writer error, not that the flight finished or that return succeeded. CSV phase `cleanup` identifies host cleanup and may include return or emergency stop.

CSV time zero is the first policy observation where available. Original host timestamps and richer observations, features, commands, measured track and config remain in the raw JSON files.
