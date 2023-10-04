# DeepShallow


-----

## Comparison:
|            | Deep-Shallow  | Large-Small (Channel Sizes) | Wide-Narrow (Kernel Size) | Long-Short (Time Stamps) |
|------------|---------------|-----------------------------|---------------------------|--------------------------|
| **Structure** | Layers reduce over time. | Channels decrease over time. | Kernel size reduces over time. | Time stamps reduce over time. |
| **Effects** | Detailed features captured early, simpler features later. | Complex patterns captured early, simpler patterns later. | Macro patterns captured early, micro patterns later. | General patterns captured early, specific patterns later. |
| **Logic** | Utilizes decreasing layers over time. | Uses more channels initially for complexity, fewer later for simplicity. | Large kernels for context, smaller kernels for details. | More data for context initially, less data for specificity later. |
| **Math Logic** | - | \(W_t = W_{\text{max}} - t \times \frac{W_{\text{max}} - W_{\text{min}}}{T}\) | \(K_t = K_{\text{max}} - t \times \frac{K_{\text{max}} - K_{\text{min}}}{T}\) | \(D_t = D_{\text{max}} - t \times \frac{D_{\text{max}} - D_{\text{min}}}{T}\) |

