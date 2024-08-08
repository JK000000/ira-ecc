use std::collections::HashMap;
use std::f64::consts::PI;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use peroxide::fuga::*;
use minilp::{Problem, OptimizationDirection, ComparisonOp, LinearExpr};

/**
  This module provides implementation of method 1. from "Design Methods for Irregular Repeat–Accumulate Codes"
  (IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 50, NO. 8, AUGUST 2004)
**/

struct IntegralODEProblem {
    a: f64,
    b: f64
}

impl ODEProblem for IntegralODEProblem {
    fn initial_conditions(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn rhs(&self, t: f64, _y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        dy[0] = (-t.powi(2)).exp() * (1.0 + (-self.a * t - self.b).exp()).log2();
        Ok(())
    }
}

/// Approximates \Int_{-\Infty}^{\Infty} e^{-z^2} log_2(1 + e^{-a*z-b}) DZ
pub(crate) fn approximate_integral(a: f64, b: f64) -> f64 {
    let ode_problem = IntegralODEProblem {
        a,
        b
    };

    let rkf = RKF45 {
        tol: 1e-6,
        safety_factor: 0.9,
        min_step_size: 1e-7,
        max_step_size: 1e-1,
        max_step_iter: 100,
    };
    let basic_ode_solver = BasicODESolver::new(rkf);
    let (_, y_vec) = basic_ode_solver.solve(
        &ode_problem,
        (-15f64, 15f64),
        0.01,
    ).unwrap();
    let y_vec: Vec<f64> = y_vec.into_iter().flatten().collect();

    y_vec.last().unwrap().clone()
}

#[derive(Clone)]
pub(crate) struct JFunc {
    data: Vec<(f64, f64)>
}


impl JFunc {

    const MAX_VAL: f64 = 100.0;
    const NUM_STEPS: usize = 500000;

    pub(crate) fn new(offset: f64) -> JFunc {

        let t: Vec<_> = (0..Self::NUM_STEPS).collect();

        let data: Vec<_> = t.into_par_iter().map(|i| {
            let mu = i as f64 * Self::MAX_VAL / Self::NUM_STEPS as f64;

            let val = approximate_integral(-2.0*mu.sqrt(), mu + offset);

            assert!(!val.is_nan());

            (mu, val)
        }).collect();

        JFunc {
            data
        }
    }

    pub(crate) fn val(&self, mu: f64) -> f64 {
        assert!(mu >= 0.0);

        if mu > Self::MAX_VAL {
            return 1.0;
        }

        let idx = (mu * Self::NUM_STEPS as f64 / Self::MAX_VAL as f64).floor() as usize;
        let offset = (mu * Self::NUM_STEPS as f64 / Self::MAX_VAL as f64).fract();

        if idx == Self::NUM_STEPS - 1 {
            return 1.0;
        }

        1.0 - (1.0/PI.sqrt()) * ( (1.0 - offset) * self.data[idx].1 + offset * self.data[idx+1].1 )
    }

    pub(crate) fn val_integral(&self, mu: f64) -> f64 {
        assert!(mu >= 0.0);

        if mu > Self::MAX_VAL {
            return 0.0;
        }

        let idx = (mu * Self::NUM_STEPS as f64 / Self::MAX_VAL as f64).floor() as usize;
        let offset = (mu * Self::NUM_STEPS as f64 / Self::MAX_VAL as f64).fract();

        if idx == Self::NUM_STEPS - 1 {
            return 0.0;
        }

        (1.0 - offset) * self.data[idx].1 + offset * self.data[idx+1].1
    }

    pub(crate) fn inverse(&self, val: f64) -> f64 {

        assert!(val >= 0.0);
        assert!(val <= 1.0);

        if val == 1.0 {
            return f64::INFINITY;
        }

        if val == 0.0 {
            return 0.0;
        }

        let mut p = 0;
        let mut q = self.data.len() - 1;

        while p + 1 < q {
            let mid = (p+q) / 2;

            if 1.0 - (1.0/PI.sqrt()) * self.data[mid].1 < val {
                p = mid;
            } else {
                q = mid;
            }
        }

        let lower = 1.0 - (1.0/PI.sqrt()) * self.data[p].1;
        let upper = 1.0 - (1.0/PI.sqrt()) * self.data[q].1;

        let ratio = (upper - val) / (upper - lower);

        ratio * self.data[p].0 + (1.0 - ratio) * self.data[q].0
    }
}

struct PhiTildeSolver {
    a: f64,
    p: f64,
    jfunc: JFunc,
    jfunc_int1: JFunc,
    jfunc_int2: JFunc
}

impl PhiTildeSolver {
    fn mu_tilde(&self, x: f64, x_tilde: f64) -> f64 {
        let c1 = self.a * self.jfunc.inverse(1.0 - x);
        let c2 = self.jfunc.inverse(1.0 - x_tilde);
        let i = self.jfunc.val(c1 + c2);
        self.jfunc.inverse(1.0 - i)
    }

    fn phi_tilde(&self, x: f64, x_tilde: f64) -> f64 {

        let m = self.mu_tilde(x, x_tilde);

        let int_1 = self.jfunc_int1.val_integral(m);//approximate_integral(2.0 * m.sqrt(), m - ((1.0-self.p)/self.p).ln());
        let int_2 = self.jfunc_int2.val_integral(m);//approximate_integral(2.0 * m.sqrt(), m + ((1.0-self.p)/self.p).ln());

        let sum = (1.0 / PI.sqrt()) * ( self.p * int_1 + (1.0 - self.p) * int_2 );

        1.0 - sum
    }

    // For given x, approximates x_tilde such that x_tilde = phi_tilde(x, x_tilde)
    fn solve(&self, x: f64) -> f64 {
        let mut p = 0.0000001_f64;
        let mut e = 0.9999999_f64;

        while (p-e).abs() > 1e-10 {
            let mid = (p + e) / 2.0;

            let val = self.phi_tilde(x, mid);

            if val < mid {
                e = mid;
            } else {
                p = mid;
            }
        }

        (p + e) / 2.0
    }
}

struct PhiFuncCoefficients {
    a: f64,
    p: f64,
    jfunc: JFunc,
    jfunc_int1: JFunc,
    jfunc_int2: JFunc,
    solver: PhiTildeSolver
}

impl PhiFuncCoefficients {

    fn new(a: f64, p: f64) -> PhiFuncCoefficients {
        let jfunc = JFunc::new(0.0);
        let jfunc_int1 = JFunc::new(- ((1.0-p)/p).ln());
        let jfunc_int2 = JFunc::new(((1.0-p)/p).ln());

        PhiFuncCoefficients {
            a,
            p,
            jfunc: jfunc.clone(),
            jfunc_int1: jfunc_int1.clone(),
            jfunc_int2: jfunc_int2.clone(),
            solver: PhiTildeSolver {
                a,
                p,
                jfunc,
                jfunc_int1,
                jfunc_int2
            }
        }
    }

    fn mu(&self, x: f64, x_tilde: f64) -> f64 {
        let c1 = (self.a - 1.0) * self.jfunc.inverse(1.0 - x);
        let c2 = 2.0 * self.jfunc.inverse(1.0 - x_tilde);
        let i = self.jfunc.val(c1 + c2);
        let res = self.jfunc.inverse(1.0 - i);

        res
    }

    fn coefficient(&self, x: f64, i: usize) -> f64 {

        let m = self.mu(x, self.solver.solve(x));

        let int_1 = self.jfunc_int1.val_integral((i-1) as f64 * m);//approximate_integral(2.0 * ( (i-1) as f64 * m ), (i-1) as f64 * m - ((1.0-self.p)/self.p).ln());
        let int_2 = self.jfunc_int2.val_integral((i-1) as f64 * m);//approximate_integral(2.0 * ( (i-1) as f64 * m ), (i-1) as f64 * m + ((1.0-self.p)/self.p).ln());
        (1.0 / PI.sqrt()) * ( self.p * int_1 + (1.0 - self.p) * int_2 )
    }
}

pub fn solve_optimal_degree_distribution(a: usize, p: f64, d: usize, num_x_steps: usize) -> HashMap<usize, f64> {

    let coefficients = PhiFuncCoefficients::new(a as f64, p);

    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let mut vars = Vec::new();
    let mut vars_sum = LinearExpr::empty();

    for i in 2..=d {
        let var = problem.add_var(1.0 / (i as f64), (0.0, f64::INFINITY));
        vars.push(var);
        vars_sum.add(var, 1.0);
    }


    problem.add_constraint(vars_sum, ComparisonOp::Eq, 1.0);

    for x_step in 0..num_x_steps {
        let x = (x_step as f64 + 0.5) / num_x_steps as f64;

        let mut expr = LinearExpr::empty();

        for i in 2..=d {
            let c = coefficients.coefficient(x, i);
            assert!(!c.is_nan());
            expr.add(vars[i-2], c);
        }

        problem.add_constraint(expr, ComparisonOp::Le, 1.0 - x);

    }

    let solution = problem.solve().unwrap();

    let mut res = Vec::new();

    for i in 2..=d {
        let val = solution[vars[i-2]];

        if val != 0.0 {
            res.push((i, val));
        }
    }

    HashMap::from_iter(res.into_iter())
}