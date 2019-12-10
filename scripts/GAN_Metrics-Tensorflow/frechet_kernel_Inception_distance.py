import tensorflow as tf
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from scipy import misc

tfgan = tf.contrib.gan

session = tf.InteractiveSession()

def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  # Unlike numpy, tensorflow's return order is (s, u, v)
  s, u, v = linalg_ops.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = array_ops.where(math_ops.less(s, eps), s, math_ops.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return math_ops.matmul(
      math_ops.matmul(u, array_ops.diag(si)), v, transpose_b=True)

def trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = math_ops.matmul(sqrt_sigma,
                                    math_ops.matmul(sigma_v, sqrt_sigma))

  return math_ops.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):
    """Classifier distance for evaluating a generative model.

    This methods computes the Frechet classifier distance from activations of
    real images and generated images. This can be used independently of the
    frechet_classifier_distance() method, especially in the case of using large
    batches during evaluation where we would like precompute all of the
    activations before computing the classifier distance.

    This technique is described in detail in https://arxiv.org/abs/1706.08500.
    Given two Gaussian distribution with means m and m_w and covariance matrices
    C and C_w, this function calculates

                  |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

    which captures how different the distributions of real images and generated
    images (or more accurately, their visual features) are. Note that unlike the
    Inception score, this is a true distance and utilizes information about real
    world images.

    Note that when computed using sample means and sample covariance matrices,
    Frechet distance is biased. It is more biased for small sample sizes. (e.g.
    even if the two distributions are the same, for a small sample size, the
    expected Frechet distance is large). It is important to use the same
    sample size to compute frechet classifier distance when comparing two
    generative models.

    Args:
      real_activations: 2D Tensor containing activations of real data. Shape is
        [batch_size, activation_size].
      generated_activations: 2D Tensor containing activations of generated data.
        Shape is [batch_size, activation_size].

    Returns:
     The Frechet Inception distance. A floating-point scalar of the same type
     as the output of the activations.

    """
    real_activations.shape.assert_has_rank(2)
    generated_activations.shape.assert_has_rank(2)

    activations_dtype = real_activations.dtype
    if activations_dtype != dtypes.float64:
        real_activations = math_ops.to_double(real_activations)
        generated_activations = math_ops.to_double(generated_activations)

    # Compute mean and covariance matrices of activations.
    m = math_ops.reduce_mean(real_activations, 0)
    m_w = math_ops.reduce_mean(generated_activations, 0)
    num_examples_real = math_ops.to_double(array_ops.shape(real_activations)[0])
    num_examples_generated = math_ops.to_double(
        array_ops.shape(generated_activations)[0])

    # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
    real_centered = real_activations - m
    sigma = math_ops.matmul(
        real_centered, real_centered, transpose_a=True) / (
                    num_examples_real - 1)

    gen_centered = generated_activations - m_w
    sigma_w = math_ops.matmul(
        gen_centered, gen_centered, transpose_a=True) / (
                      num_examples_generated - 1)

    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = math_ops.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    # Next the distance between means.
    mean = math_ops.reduce_sum(
        math_ops.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
    fid = trace + mean
    if activations_dtype != dtypes.float64:
        fid = math_ops.cast(fid, activations_dtype)

    return fid

def kernel_classifier_distance_and_std_from_activations(real_activations,
                                                        generated_activations,
                                                        max_block_size=10,
                                                        dtype=None):
    """Kernel "classifier" distance for evaluating a generative model.

    This methods computes the kernel classifier distance from activations of
    real images and generated images. This can be used independently of the
    kernel_classifier_distance() method, especially in the case of using large
    batches during evaluation where we would like to precompute all of the
    activations before computing the classifier distance, or if we want to
    compute multiple metrics based on the same images. It also returns a rough
    estimate of the standard error of the estimator.

    This technique is described in detail in https://arxiv.org/abs/1801.01401.
    Given two distributions P and Q of activations, this function calculates

        E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
          - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]

    where k is the polynomial kernel

        k(x, y) = ( x^T y / dimension + 1 )^3.

    This captures how different the distributions of real and generated images'
    visual features are. Like the Frechet distance (and unlike the Inception
    score), this is a true distance and incorporates information about the
    target images. Unlike the Frechet score, this function computes an
    *unbiased* and asymptotically normal estimator, which makes comparing
    estimates across models much more intuitive.

    The estimator used takes time quadratic in max_block_size. Larger values of
    max_block_size will decrease the variance of the estimator but increase the
    computational cost. This differs slightly from the estimator used by the
    original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.
    The estimate of the standard error will also be more reliable when there are
    more blocks, i.e. when max_block_size is smaller.

    NOTE: the blocking code assumes that real_activations and
    generated_activations are both in random order. If either is sorted in a
    meaningful order, the estimator will behave poorly.

    Args:
      real_activations: 2D Tensor containing activations of real data. Shape is
        [batch_size, activation_size].
      generated_activations: 2D Tensor containing activations of generated data.
        Shape is [batch_size, activation_size].
      max_block_size: integer, default 1024. The distance estimator splits samples
        into blocks for computational efficiency. Larger values are more
        computationally expensive but decrease the variance of the distance
        estimate. Having a smaller block size also gives a better estimate of the
        standard error.
      dtype: if not None, coerce activations to this dtype before computations.

    Returns:
     The Kernel Inception Distance. A floating-point scalar of the same type
       as the output of the activations.
     An estimate of the standard error of the distance estimator (a scalar of
       the same type).
    """

    real_activations.shape.assert_has_rank(2)
    generated_activations.shape.assert_has_rank(2)
    real_activations.shape[1].assert_is_compatible_with(
        generated_activations.shape[1])

    if dtype is None:
        dtype = real_activations.dtype
        assert generated_activations.dtype == dtype
    else:
        real_activations = math_ops.cast(real_activations, dtype)
        generated_activations = math_ops.cast(generated_activations, dtype)

    # Figure out how to split the activations into blocks of approximately
    # equal size, with none larger than max_block_size.
    n_r = array_ops.shape(real_activations)[0]
    n_g = array_ops.shape(generated_activations)[0]

    n_bigger = math_ops.maximum(n_r, n_g)
    n_blocks = math_ops.to_int32(math_ops.ceil(n_bigger / max_block_size))

    v_r = n_r // n_blocks
    v_g = n_g // n_blocks

    n_plusone_r = n_r - v_r * n_blocks
    n_plusone_g = n_g - v_g * n_blocks

    sizes_r = array_ops.concat([
        array_ops.fill([n_blocks - n_plusone_r], v_r),
        array_ops.fill([n_plusone_r], v_r + 1),
    ], 0)
    sizes_g = array_ops.concat([
        array_ops.fill([n_blocks - n_plusone_g], v_g),
        array_ops.fill([n_plusone_g], v_g + 1),
    ], 0)

    zero = array_ops.zeros([1], dtype=dtypes.int32)
    inds_r = array_ops.concat([zero, math_ops.cumsum(sizes_r)], 0)
    inds_g = array_ops.concat([zero, math_ops.cumsum(sizes_g)], 0)

    dim = math_ops.cast(tf.shape(real_activations)[1], dtype)

    def compute_kid_block(i):
        'Compute the ith block of the KID estimate.'
        r_s = inds_r[i]
        r_e = inds_r[i + 1]
        r = real_activations[r_s:r_e]
        m = math_ops.cast(r_e - r_s, dtype)

        g_s = inds_g[i]
        g_e = inds_g[i + 1]
        g = generated_activations[g_s:g_e]
        n = math_ops.cast(g_e - g_s, dtype)

        k_rr = (math_ops.matmul(r, r, transpose_b=True) / dim + 1)**3
        k_rg = (math_ops.matmul(r, g, transpose_b=True) / dim + 1)**3
        k_gg = (math_ops.matmul(g, g, transpose_b=True) / dim + 1)**3
        return (-2 * math_ops.reduce_mean(k_rg) +
                (math_ops.reduce_sum(k_rr) - math_ops.trace(k_rr)) / (m * (m - 1)) +
                (math_ops.reduce_sum(k_gg) - math_ops.trace(k_gg)) / (n * (n - 1)))

    ests = functional_ops.map_fn(
        compute_kid_block, math_ops.range(n_blocks), dtype=dtype, back_prop=False)

    mn = math_ops.reduce_mean(ests)

    # nn_impl.moments doesn't use the Bessel correction, which we want here
    n_blocks_ = math_ops.cast(n_blocks, dtype)
    var = control_flow_ops.cond(
        math_ops.less_equal(n_blocks, 1),
        lambda: array_ops.constant(float('nan'), dtype=dtype),
        lambda: math_ops.reduce_sum(math_ops.square(ests - mn)) / (n_blocks_ - 1))

    return mn, math_ops.sqrt(var / n_blocks_)


def inception_activations(images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    activations = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='pool_3:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


def get_inception_activations(batch_size, images, inception_images, activations):
    n_batches = images.shape[0] // batch_size
    act = np.zeros([n_batches * batch_size, 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = images[i * batch_size:(i + 1) * batch_size] / 255. * 2 - 1
        act[i * batch_size:(i + 1) * batch_size] = activations.eval(feed_dict={inception_images: inp})
    return act


def activations2distance(fcd, real_activation, fake_activation, act1, act2):
    return fcd.eval(feed_dict={real_activation: act1, fake_activation: act2})


def get_fid(fcd, batch_size, images1, images2, inception_images, real_activation, fake_activation, activations):
    # print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(batch_size, images1, inception_images, activations)
    act2 = get_inception_activations(batch_size, images2, inception_images, activations)
    fid = activations2distance(fcd, real_activation, fake_activation, act1, act2)
    # print('FID calculation time: %f s' % (time.time() - start_time))
    return fid

def get_kid(kcd, batch_size, images1, images2, inception_images, real_activation, fake_activation, activations):
    # print('Calculating KID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(batch_size, images1, inception_images, activations)
    act2 = get_inception_activations(batch_size, images2, inception_images, activations)
    kcd = activations2distance(kcd, real_activation, fake_activation, act1, act2)
    # print('KID calculation time: %f s' % (time.time() - start_time))
    return kcd

def get_images(filename):
    x = misc.imread(filename)
    x = misc.imresize(x, size=[299, 299])
    return x


