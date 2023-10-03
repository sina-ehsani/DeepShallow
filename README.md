# DeepShallow


-----

## Comparison:
|            | Deep-Shallow | Large-Small (Channel Sizes) | Wide-Narrow (Kernel Size) | Long-Short (Time Stamps) |
|------------|--------------|-----------------------------|---------------------------|--------------------------|
| **Structure** | Depth decreases over time. | More channels early on, fewer later. | Large kernels initially, smaller later. | More timestamps initially, fewer later. |
| **Effects** | Early layers capture broad trends; later layers focus on recent changes. | Early stages capture complex patterns; later stages model direct relationships. | Early stages capture macro patterns; later stages focus on micro patterns. | Early stages for overall patterns; later stages for specific, recent trends. |
| **Logic** | Utilizes backpropagation's nature of capturing detailed features in earlier layers. | More channels equate to complex feature extraction initially; fewer channels for simplified, focused features later. | Broad view initially for overall context; focused view later for detailed, localized patterns. | Captures broader temporal context initially; focuses on recent, more relevant data later. |


