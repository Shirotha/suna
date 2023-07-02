use num_traits::{cast::{cast, AsPrimitive}, Float, Inv, PrimInt};
use ordered_float::NotNan as NonNaN;
use duplicate::duplicate_item;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    InputIdentity(usize),
    InputSigmoid(usize),
    OutputIdentity(usize),
    OutputSigmoid(usize),
    Control,
    Identity,
    Sigmoid,
    Threshold,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Speed(u32);
impl<T: AsPrimitive<u32>> From<T> for Speed {
    fn from(value: T) -> Self { 
        let speed = value.as_();
        if speed == 0 { panic!("Speed cannot be 0") }
        Speed(speed)
    }
}
#[duplicate_item(fX; [f32]; [f64])]
impl From<Speed> for NonNaN<fX> {
    fn from(value: Speed) -> Self { NonNaN::new((value.0 as fX).inv()).unwrap() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuronGene {
    pub(crate) id: usize,
    pub(crate) speed: Speed,
    pub(crate) r#type: Type,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Weight<T: Float> {
    Intrinsic(NonNaN<T>),
    Modulated(usize),
}

pub struct ConnectionGene<T: Float> {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) weight: Weight<T>,
}

pub struct Genome<T: Float> {
    pub(crate) neurons: Vec<NeuronGene>,
    pub(crate) connections: Vec<ConnectionGene<T>>,
}
impl<T: Float> Genome<T> {
    pub fn new(inputs: usize, smooth_input: bool, outputs: usize, smooth_output: bool) -> Self {
        let mut neurons = Vec::with_capacity(inputs + outputs);
        for i in 0..inputs {
            let r#type = if smooth_input { Type::InputSigmoid(i) } else { Type::InputIdentity(i) };
            neurons.push(NeuronGene { id: i + 1, speed: 1.into(), r#type })
        }
        for i in 0..outputs {
            let r#type = if smooth_output { Type::OutputSigmoid(i) } else { Type::OutputIdentity(i) };
            neurons.push(NeuronGene { id: i + inputs + 1, speed: 1.into(), r#type })
        }
        Genome { neurons, connections: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spectrum<T: PrimInt> {
    count: T,
    identity: T,
    sigmoid: T,
    threshold: T,
    random: T,
    control: T,
    slow: T,
}
impl<T: PrimInt> Spectrum<T> {
    pub fn new<F: Float>(genome: &Genome<F>) -> Self {
        let mut count = genome.neurons.len();
        let mut identity = 0;
        let mut sigmoid = 0;
        let mut threshold = 0;
        let mut random = 0;
        let mut control = 0;
        let mut slow = 0;
        
        for neuron in genome.neurons.iter() {
            if neuron.speed.0 > 1 { slow += 1; }
            match neuron.r#type {
                Type::Identity => identity += 1,
                Type::Sigmoid => sigmoid += 1,
                Type::Threshold => threshold += 1,
                Type::Random => random += 1,
                Type::Control => control += 1,
                _ => count -= 1,
            }
        }
        Spectrum {
            count: cast(count).unwrap(),
            identity: cast(identity).unwrap(),
            sigmoid: cast(sigmoid).unwrap(),
            threshold: cast(threshold).unwrap(),
            random: cast(random).unwrap(),
            control: cast(control).unwrap(),
            slow: cast(slow).unwrap(),
        }
    }
    #[inline] fn int_dist<F: Float>(a: T, b: T) -> F {
        let d = if a < b { b - a } else { a - b };
        cast(d).unwrap()
    }
    pub fn distance_absolute<F: Float>(&self, other: &Spectrum<T>) -> NonNaN<F> {
        let identity = Self::int_dist::<F>(self.identity, other.identity);
        let sigmoid = Self::int_dist::<F>(self.sigmoid, other.sigmoid);
        let threshold = Self::int_dist::<F>(self.threshold, other.threshold);
        let random = Self::int_dist::<F>(self.random, other.random);
        let control = Self::int_dist::<F>(self.control, other.control);
        let slow = Self::int_dist::<F>(self.slow, other.slow);
        NonNaN::new((identity * identity
            + sigmoid * sigmoid
            + threshold * threshold
            + random * random
            + control * control
            + slow * slow).sqrt()).unwrap()
    }
    #[inline] fn ratio_dist<F: Float>(a: T, n_a: T, b: T, n_b: T) -> F {
        let a = if n_a.is_zero() { F::zero() } else { cast::<T, F>(a).unwrap() / cast::<T, F>(n_a).unwrap() };
        let b = if n_b.is_zero() { F::zero() } else { cast::<T, F>(b).unwrap() / cast::<T, F>(n_b).unwrap() };
        (a - b).abs()
    }
    pub fn distance_ratio<F: Float>(&self, other: &Spectrum<T>) -> NonNaN<F> {
        let n = self.count; let m = other.count;
        let identity = Self::ratio_dist::<F>(self.identity, n, other.identity, m);
        let sigmoid = Self::ratio_dist::<F>(self.sigmoid, n, other.sigmoid, m);
        let threshold = Self::ratio_dist::<F>(self.threshold, n, other.threshold, m);
        let random = Self::ratio_dist::<F>(self.random, n, other.random, m);
        let control = Self::ratio_dist::<F>(self.control, n, other.control, m);
        let slow = Self::ratio_dist::<F>(self.slow, n, other.slow, m);
        NonNaN::new((identity * identity
            + sigmoid * sigmoid
            + threshold * threshold
            + random * random
            + control * control
            + slow * slow).sqrt()).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num_traits::Zero;
    use ordered_float::NotNan as NonNaN;
    use duplicate::duplicate;

    #[test]
    fn speed_conversion() {
        duplicate!{[
            int_type    float_type;
            [i32]       [f32];
            [i64]       [f32];
            [u32]       [f32];
            [u64]       [f32];
            [usize]     [f32];
            [i32]       [f64];
            [i64]       [f64];
            [u32]       [f64];
            [u64]       [f64];
            [usize]     [f64];
        ]
            let speed: int_type = 2;
            let speed: Speed = speed.into();
            let speed: NonNaN<float_type> = speed.into();
            assert_eq!(speed, 0.5);
        }
    }

    #[test]
    #[allow(clippy::nonminimal_bool, clippy::identity_op)]
    fn create_empty_genome() {
        duplicate!{[
            inputs  smooth_input    outputs smooth_output;
            [2]     [false]         [2]     [false];
            [2]     [true]          [2]     [true];
            [0]     [false]         [1]     [false];
            [2]     [false]         [0]     [false];
        ]
            let genome: Genome<f32> = Genome::new(inputs, smooth_input, outputs, smooth_output);
            assert_eq!(genome.neurons.len(), inputs + outputs);
            for (i, neuron) in genome.neurons.iter().take(inputs).enumerate() {
                assert_eq!(neuron.id, i + 1);
                let speed: NonNaN<f32> = neuron.speed.into();
                assert_eq!(speed, 1.0);
                match neuron.r#type {
                    Type::InputIdentity(slot) => { if smooth_input { panic!("wrong neuron type"); } assert_eq!(slot, i); }
                    Type::InputSigmoid(slot) => { if !smooth_input { panic!("wrong neuron type"); } assert_eq!(slot, i); }
                    _ => panic!("wrong neuron type")
                }
            }
            for (i, neuron) in genome.neurons.iter().skip(inputs).enumerate() {
                assert_eq!(neuron.id, inputs + i + 1);
                let speed: NonNaN<f32> = neuron.speed.into();
                assert_eq!(speed, 1.0);
                match neuron.r#type {
                    Type::OutputIdentity(slot) => { if smooth_output { panic!("wrong neuron type"); } assert_eq!(slot, i); }
                    Type::OutputSigmoid(slot) => { if !smooth_output { panic!("wrong neuron type"); } assert_eq!(slot, i); }
                    _ => panic!("wrong neuron type")
                }
            }
            assert!(genome.connections.is_empty());
        }
    }

    #[test]
    fn compare_spectrum() {
        let mut genome: Genome<f32> = Genome::new(2, false, 2, false);
        let spectrum: Spectrum<usize> = Spectrum { count: 10, identity: 3, sigmoid: 2, threshold: 1, random: 1, control: 3, slow: 5 };
        let mut id = genome.neurons.len() + 1;
        let mut slow = spectrum.slow;
        for _ in 0..spectrum.identity {
            let speed = if slow > 0 { slow -= 1; 7.into() } else { 1.into() };
            genome.neurons.push(NeuronGene { id, speed, r#type: Type::Identity });
            id += 1;
        }
        for _ in 0..spectrum.sigmoid {
            let speed = if slow > 0 { slow -= 1; 7.into() } else { 1.into() };
            genome.neurons.push(NeuronGene { id, speed, r#type: Type::Sigmoid });
            id += 1;
        }
        for _ in 0..spectrum.threshold {
            let speed = if slow > 0 { slow -= 1; 7.into() } else { 1.into() };
            genome.neurons.push(NeuronGene { id, speed, r#type: Type::Threshold });
            id += 1;
        }
        for _ in 0..spectrum.random {
            let speed = if slow > 0 { slow -= 1; 7.into() } else { 1.into() };
            genome.neurons.push(NeuronGene { id, speed, r#type: Type::Random });
            id += 1;
        }
        for _ in 0..spectrum.control {
            let speed = if slow > 0 { slow -= 1; 7.into() } else { 1.into() };
            genome.neurons.push(NeuronGene { id, speed, r#type: Type::Control });
            id += 1;
        }
        let genome = Spectrum::new(&genome);
        assert_eq!(genome, spectrum);
        assert_eq!(genome.distance_absolute::<f32>(&spectrum), 0.0);
        assert_eq!(genome.distance_ratio::<f32>(&spectrum), 0.0);

        let spectrum: Spectrum<usize> = Spectrum { count: 7, identity: 2, sigmoid: 2, threshold: 1, random: 1, control: 1, slow: 2 };
        assert_ne!(genome, spectrum);
        assert!(genome.distance_absolute::<f32>(&spectrum) > NonNaN::zero());
        assert!(genome.distance_ratio::<f32>(&spectrum) > NonNaN::zero());
    }
}