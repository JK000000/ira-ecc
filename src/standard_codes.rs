use std::collections::HashMap;
use crate::{Bit, DecodingOptions, DecodingResult, EccCode, IraEccCode, IraEccCodeCompressedRepresentation};


/// Message size: 0.25kb (2048 bit) <br>
/// Encoded size: 32kb (262144 bit) <br>
/// <br>
/// Performance: <br>
/// Binary Symmetric Channel: Up to 38% bit flip probability <br>
/// Binary Erasure Channel: Up to 96% bit erasure probability <br>
/// Additive White Gaussian Noise channel: Down to 0.065 signal-to-noise ratio <br>
pub struct IraCode1128 {
    code: IraEccCode,
}

impl crate::standard_codes::IraCode1128 {
    pub fn new() -> crate::standard_codes::IraCode1128 {
        let compressed_repr = include_bytes!("../res/code1128.bin");
        let repr = IraEccCodeCompressedRepresentation::decompress(compressed_repr);

        crate::standard_codes::IraCode1128 {
            code: IraEccCode::from_representation(&repr)
        }
    }

    pub fn decoding_options() -> DecodingOptions {
        DecodingOptions {
            min_rounds: 5,
            max_rounds: 50,
            max_message: 30.0,
            epsilon: 1e-1,
        }
    }
}

impl EccCode for IraCode1128 {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        self.code.decode(&data[..self.code.encoded_len()], options)
    }

    fn encode(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.encoded_len());
        self.code.encode(&mut data[..self.code.encoded_len()]);
    }

    fn message_len(&self) -> usize {
        2048
    }

    fn encoded_len(&self) -> usize {
        262144
    }
}


/// Message size: 1kb (8192 bit) <br>
/// Encoded size: 16kb (131072 bit) <br>
/// <br>
/// Performance: <br>
/// Binary Symmetric Channel: Up to 29% bit flip probability <br>
/// Binary Erasure Channel: Up to 88% bit erasure probability <br>
/// Additive White Gaussian Noise channel: Down to 0.19 signal-to-noise ratio <br>
pub struct IraCode116 {
    code: IraEccCode,
}

impl IraCode116 {
    pub fn new() -> IraCode116 {
        let compressed_repr = include_bytes!("../res/code116.bin");
        let repr = IraEccCodeCompressedRepresentation::decompress(compressed_repr);

        IraCode116 {
            code: IraEccCode::from_representation(&repr)
        }
    }

    pub fn decoding_options() -> DecodingOptions {
        DecodingOptions {
            min_rounds: 5,
            max_rounds: 50,
            max_message: 30.0,
            epsilon: 1e-1,
        }
    }
}

impl EccCode for IraCode116 {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        self.code.decode(&data[..self.code.encoded_len()], options)
    }

    fn encode(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.encoded_len());
        self.code.encode(&mut data[..self.code.encoded_len()]);
    }

    fn message_len(&self) -> usize {
        8192
    }

    fn encoded_len(&self) -> usize {
        131072
    }
}


/// Message size: 4kb (32768 bit) <br>
/// Encoded size: 32kb (262144 bit) <br>
/// <br>
/// Performance: <br>
/// Binary Symmetric Channel: Up to 23% bit flip probability <br>
/// Binary Erasure Channel: Up to 78% bit erasure probability <br>
/// Additive White Gaussian Noise channel: Down to 0.415 signal-to-noise ratio
pub struct IraCode18 {
    code: IraEccCode,
}

impl IraCode18 {
    pub fn new() -> IraCode18 {
        let compressed_repr = include_bytes!("../res/code18.bin");
        let repr = IraEccCodeCompressedRepresentation::decompress(compressed_repr);

        IraCode18 {
            code: IraEccCode::from_representation(&repr)
        }
    }

    pub fn decoding_options() -> DecodingOptions {
        DecodingOptions {
            min_rounds: 5,
            max_rounds: 50,
            max_message: 30.0,
            epsilon: 1e-1,
        }
    }
}

impl EccCode for IraCode18 {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        self.code.decode(&data[..self.code.encoded_len()], options)
    }

    fn encode(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.encoded_len());
        self.code.encode(&mut data[..self.code.encoded_len()]);
    }

    fn message_len(&self) -> usize {
        32768
    }

    fn encoded_len(&self) -> usize {
        262144
    }
}


/// Message size: 4kb (32768 bit) <br>
/// Encoded size: 16kb (131072 bit) <br>
/// <br>
/// Performance: <br>
/// Binary Symmetric Channel: Up to 17.6% bit flip probability <br>
/// Binary Erasure Channel: Up to 65.8% bit erasure probability <br>
/// Additive White Gaussian Noise channel: Down to 0.68 signal-to-noise ratio <br>
pub struct IraCode14 {
    code: IraEccCode,
}

impl IraCode14 {
    pub fn new() -> IraCode14 {
        let compressed_repr = include_bytes!("../res/code14.bin");
        let repr = IraEccCodeCompressedRepresentation::decompress(compressed_repr);

        IraCode14 {
            code: IraEccCode::from_representation(&repr)
        }
    }

    pub fn decoding_options() -> DecodingOptions {
        DecodingOptions {
            min_rounds: 5,
            max_rounds: 50,
            max_message: 30.0,
            epsilon: 1e-2,
        }
    }
}

impl EccCode for IraCode14 {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        self.code.decode(&data[..self.code.encoded_len()], options)
    }

    fn encode(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.encoded_len());
        self.code.encode(&mut data[..self.code.encoded_len()]);
    }

    fn message_len(&self) -> usize {
        32768
    }

    fn encoded_len(&self) -> usize {
        131072
    }
}


/// Message size: 8kb (65536 bit) <br>
/// Encoded size: 16kb (131072 bit) <br>
/// <br>
/// Performance: <br>
/// Binary Symmetric Channel: Up to 6.5% bit flip probability <br>
/// Binary Erasure Channel: Up to 37.2% bit erasure probability <br>
/// Additive White Gaussian Noise channel: Down to 1.73 signal-to-noise ratio <br>
pub struct IraCode12 {
    code: IraEccCode,
}

impl IraCode12 {
    pub fn new() -> IraCode12 {
        let compressed_repr = include_bytes!("../res/code12.bin");
        let repr = IraEccCodeCompressedRepresentation::decompress(compressed_repr);

        IraCode12 {
            code: IraEccCode::from_representation(&repr)
        }
    }

    pub fn decoding_options() -> DecodingOptions {
        DecodingOptions {
            min_rounds: 5,
            max_rounds: 50,
            max_message: 30.0,
            epsilon: 1e-1,
        }
    }
}

impl EccCode for IraCode12 {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        self.code.decode(data, options)
    }

    fn encode(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.encoded_len());
        self.code.encode(data);
    }

    fn message_len(&self) -> usize {
        65536
    }

    fn encoded_len(&self) -> usize {
        131072
    }
}


/// Message size: 4kb (32768 bit) <br>
/// Encoded size: 6kb (49152 bit) <br>
/// <br>
/// Performance: <br>
/// Binary Symmetric Channel: Up to 3.75% bit flip probability <br>
/// Binary Erasure Channel: Up to 15% bit erasure probability <br>
///   In the range of 15-23% bit erasure probability the code works approximately 0.998% of times, <br>
///   and if it fails, it is often the case that only single bit is incorrect, and it is often within 10 most uncertain bits <br>
/// Additive White Gaussian Noise channel: Down to 4.7 signal-to-noise ratio
pub struct IraCode23 {
    code: IraEccCode,
}

impl IraCode23 {
    pub fn new() -> crate::standard_codes::IraCode23 {
        let compressed_repr = include_bytes!("../res/code23.bin");
        let repr = IraEccCodeCompressedRepresentation::decompress(compressed_repr);

        crate::standard_codes::IraCode23 {
            code: IraEccCode::from_representation(&repr)
        }
    }

    pub fn decoding_options() -> DecodingOptions {
        DecodingOptions {
            min_rounds: 5,
            max_rounds: 50,
            max_message: 30.0,
            epsilon: 1e-1,
        }
    }
}

impl EccCode for IraCode23 {
    fn decode(&mut self, data: &[f64], options: DecodingOptions) -> DecodingResult {
        self.code.decode(data, options)
    }

    fn encode(&self, data: &mut [Bit]) {
        assert!(data.len() >= self.encoded_len());
        self.code.encode(data);
    }

    fn message_len(&self) -> usize {
        32768
    }

    fn encoded_len(&self) -> usize {
        49152
    }
}


/// Not a generating function in mathematical meaning, just a function used to generate the ratios
fn ratios_generating_func(x: f64, mean: f64, std: f64) -> f64 {
    let mx = x - mean;

    let var = std.powi(2);

    if mx < -(2.0 * var).sqrt() {
        return 0.0;
    }

    (2.0 / (2.0 * var).sqrt()).powi(2) * (mx + (2.0 * var).sqrt()) *
        (-(2.0 / (2.0 * var).sqrt()) * (mx + (2.0 * var).sqrt())).exp()
}


pub fn generate_ratios(mean: f64, std: f64) -> HashMap<usize, f64> {

    let mut res: Vec<(usize, f64)> = vec![];

    let begin = (mean - 3.0 * std).floor() as usize;
    let end = (mean + 6.0 * std).ceil() as usize;

    let mut sum = 0.0;

    for i in begin..end {
        let val = ratios_generating_func(i as f64, mean, std);
        if val > 1e-5 {
            sum += val;
            res.push((i, val));
        }
    }

    res.iter_mut().for_each(|(_, val)| {
        *val /= sum;
    });

    HashMap::from_iter(res.into_iter())
}


pub fn edge_ratios_to_node_ratios(edge_ratios: HashMap<usize, f64>) -> HashMap<usize, f64> {
    let mut norm = 0.0;

    for (&i, &val) in edge_ratios.iter() {
        norm += val / i as f64;
    }

    HashMap::from_iter(edge_ratios.into_iter().map(|(i, val)| {
        (i, (val / i as f64) / norm)
    }))
}
