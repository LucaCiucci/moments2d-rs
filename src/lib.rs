/*!
Tools to compute moments of polygons.
*/
/// Alias for an N x N matrix of f64.
pub type Moments<const N: usize, T> = [[T; N]; N];

use num_traits::{Float, NumAssignOps};

#[derive(Debug, Clone, Copy)]
pub struct Point2d<T> {
    x: T,
    y: T,
}

impl<T> From<(T, T)> for Point2d<T> {
    fn from((x, y): (T, T)) -> Self {
        Self { x, y }
    }
}

impl<T> From<Point2d<T>> for (T, T) {
    fn from(point: Point2d<T>) -> Self {
        (point.x, point.y)
    }
}


/// Pair of points that define a segment.
pub type Segment<T> = (Point2d<T>, Point2d<T>);

fn closed_polygon_segments<T>(
    points: impl Iterator<Item = Point2d<T>> + Clone,
) -> impl Iterator<Item = Segment<T>> {
    let points_shifted = points.clone().skip(1).chain(points.clone().take(1));
    points.zip(points_shifted)
}

/// Same as [compute_moments_0], but returns a float.
pub fn area<T: Float + NumAssignOps>(points: impl Iterator<Item = Point2d<T>> + Clone) -> T {
    compute_moments_0(points)[0][0]
}

/// Computes the barycenter of a polygon.
pub fn barycenter<T: Float + NumAssignOps>(points: impl Iterator<Item = Point2d<T>> + Clone) -> (T, T) {
    let moments = compute_moments_1(points);
    let area = moments[0][0];
    let x = moments[1][0] / area;
    let y = moments[0][1] / area;
    (x, y)
}

/// Computes the first moments of a polygon.
///
/// See [compute_moments_1] for more information, this computers only the area.
pub fn compute_moments_0<T: Float + NumAssignOps>(points: impl Iterator<Item = Point2d<T>> + Clone) -> Moments<1, T> {
    let mut moments: Moments<1, T> = std::array::from_fn(|_| std::array::from_fn(|_| T::zero()));
    for segment in closed_polygon_segments(points) {
        let (p0, p1): Segment<T> = segment;

        let (x0, y0) = (p0.x, p0.y);
        let (x1, y1) = (p1.x, p1.y);


        moments[0][0] += T::from(0.5).unwrap() * (x0 * y1 - x1 * y0); // TODO replace 
    }

    moments
}

/// Computes the first moments of a polygon.
///
/// Given a polygon that defines a closed path <i-math>\partial \Sigma</i-math> containing the
/// region <i-math>\Sigma</i-math>, we define the moment <i-math>M_{ij}</i-math> as the integral:
/// <tex-math>
///     M_{ij} = \int_{\Sigma} x^i y^j d^2\Sigma
/// </tex-math>
/// just like for the [OpenCV Moments](https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html) struct.
///
/// # Computation
///
/// The theory behind the computation of the moments is described in the `/resources/notebooks/polygons/moments.ipynb` notebook in the repository of this crate.
pub fn compute_moments_1<T: Float + NumAssignOps>(points: impl Iterator<Item = Point2d<T>> + Clone) -> Moments<2, T> {
    let mut moments: Moments<2, T> = std::array::from_fn(|_| std::array::from_fn(|_| T::zero()));
    for segment in closed_polygon_segments(points) {
        let (p0, p1): Segment<T> = segment;

        let (x0, y0) = (p0.x, p0.y);
        let (x1, y1) = (p1.x, p1.y);

        let d1 = x0 * y1;
        let d2 = x1 * y0;
        let d3 = d1 - d2;
        let d4 = T::from(1.0 / 6.0).unwrap() * d3;

        moments[0][0] += T::from(0.5).unwrap() * d3;
        moments[0][1] += d4 * (y0 + y1);
        moments[1][0] += d4 * (x0 + x1);
        moments[1][1] += T::one() / T::from(24).unwrap() * d3 * (d1 + d2 + T::from(2).unwrap() * x0 * y0 + T::from(2).unwrap() * x1 * y1);
    }

    moments
}

/// Computes the first moments of a polygon.
///
/// See [compute_moments_1] for more information.
pub fn compute_moments_2<T: Float + NumAssignOps>(points: impl Iterator<Item = Point2d<T>> + Clone) -> Moments<3, T> {
    let mut moments: Moments<3, T> = std::array::from_fn(|_| std::array::from_fn(|_| T::zero()));
    for segment in closed_polygon_segments(points) {
        let (p0, p1): Segment<T> = segment;

        let (x0, y0) = (p0.x, p0.y);
        let (x1, y1) = (p1.x, p1.y);

        let d1 = x0 * y1;
        let d2 = x1 * y0;
        let d3 = d1 - d2;
        let d4 = T::one() / T::from(6).unwrap() * d3;
        let d5 = y0.powi(2);
        let d6 = y1.powi(2);
        let d7 = y0 * y1;
        let d8 = T::one() / T::from(12).unwrap() * d3;
        let d9 = T::from(2).unwrap() * y0;
        let d10 = T::from(2).unwrap() * y1;
        let d11 = d6 * x0;
        let d12 = d5 * x1;
        let d13 = T::from(3).unwrap() * x0;
        let d14 = T::from(3).unwrap() * x1;
        let d15 = T::one() / T::from(60).unwrap() * d3;
        let d16 = x0.powi(2);
        let d17 = x1.powi(2);
        let d18 = T::from(3).unwrap() * d16;
        let d19 = T::from(3).unwrap() * d17;

        moments[0][0] += T::one() / T::from(2).unwrap() * d3;
        moments[0][1] += d4 * (y0 + y1);
        moments[0][2] += d8 * (d5 + d6 + d7);
        moments[1][0] += d4 * (x0 + x1);
        moments[1][1] += T::one() / T::from(24).unwrap() * d3 * (d1 + d10 * x1 + d2 + d9 * x0);
        moments[1][2] += d15 * (d1 * d9 + d10 * d2 + d11 + d12 + d13 * d5 + d14 * d6);
        moments[2][0] += d8 * (d16 + d17 + x0 * x1);
        moments[2][1] +=
            d15 * (T::from(2).unwrap() * d1 * x1 + d16 * y1 + d17 * y0 + d18 * y0 + d19 * y1 + T::from(2).unwrap() * d2 * x0);
        moments[2][2] += T::one() / T::from(180).unwrap()
            * d3
            * (T::from(4).unwrap() * d1 * d2
                + d11 * d14
                + d12 * d13
                + T::from(6).unwrap() * d16 * d5
                + d16 * d6
                + d17 * d5
                + T::from(6).unwrap() * d17 * d6
                + d18 * d7
                + d19 * d7);
    }
    moments
}
