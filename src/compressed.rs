use std::ops::{Index, IndexMut};
use lzma::{compress, decompress};
use sha2::{Digest, Sha256};
use std::fmt::Write;

pub struct IraEccCodeCompressedRepresentation {
    pub(crate) data: Vec<usize>,
}


impl Index<usize> for IraEccCodeCompressedRepresentation {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index + 3]
    }
}

impl IndexMut<usize> for IraEccCodeCompressedRepresentation {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index + 3]
    }
}

impl IraEccCodeCompressedRepresentation {
    pub fn message_len(&self) -> usize {
        return self.data[0];
    }

    pub fn encoded_len(&self) -> usize {
        return self.data[1];
    }

    pub fn check_nodes_left_degree(&self) -> usize {
        return self.data[2];
    }

    pub fn set_message_len(&mut self, val: usize) {
        self.data[0] = val;
    }

    pub fn set_encoded_len(&mut self, val: usize) {
        self.data[1] = val;
    }

    pub fn set_check_nodes_left_degree(&mut self, val: usize) {
        self.data[2] = val;
    }

    pub fn compress(&self) -> Vec<u8> {
        let buf: Vec<u8> = self.data.iter().flat_map(|&x| x.to_le_bytes()).collect();
        compress(&buf, 9).unwrap()
    }

    pub fn decompress(buf: &[u8]) -> IraEccCodeCompressedRepresentation {
        IraEccCodeCompressedRepresentation {
            data: decompress(buf).unwrap().chunks(8).map(|x| usize::from_le_bytes(x.try_into().unwrap())).collect()
        }
    }

    pub fn checksum(&self) -> String {
        let mut hasher = Sha256::new();

        for b in self.data.iter() {
            sha2::digest::Update::update(&mut hasher, &b.to_le_bytes()[..]);
        }

        let res = hasher.finalize();

        let mut s = String::new();
        for &byte in res.as_slice().iter() {
            write!(&mut s, "{:0x}", byte).unwrap();
        }

        s
    }
}