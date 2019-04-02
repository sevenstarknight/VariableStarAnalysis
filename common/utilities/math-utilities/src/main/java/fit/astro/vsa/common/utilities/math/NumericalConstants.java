/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math;

/**
 * A collection of useful numbers (stored to maximum precision).
 * <p>
 * @version 1.0
 * @author Mark Hale
 */
public interface NumericalConstants {

    // ======================== General Constants
    /**
     * Square root of 2.
     */
    public static final double SQRT2 = 1.4142135623730950488016887242096980785696718753769;
    /**
     * Two times pi
     * <p>
     */
    public static final double TWO_PI = 6.2831853071795864769252867665590057683943387987502;

    /**
     *
     */
    public static final double FOURPISQ = TWO_PI * TWO_PI;

    /**
     * Square root of 2 pi
     */
    public static final double SQRT2PI = 2.5066282746310005024157652848110452530069867406099;
    /**
     * Euler's gamma constant.
     * <p>
     */
    public static final double GAMMA = 0.57721566490153286060651209008240243104215933593992;
    /**
     * Golden ratio.
     * <p>
     */
    public static final double GOLDEN_RATIO = 1.6180339887498948482045868343656381177203091798058;

    // ========================= Log Values
    /**
     * The natural logarithm of 10.
     * <p>
     */
    public static final double LOG10 = Math.log(10);

    /**
     * The natural logarithm of 2.
     * <p>
     */
    public static final double LOG2 = Math.log(2);

    /**
     * The natural logarithm of the maximum double value: log(MAX_VALUE).
     * <p>
     */
    public static final double MAX_LOG = Math.log(Double.MAX_VALUE);

    /**
     * The natural logarithm of the minimum double value: log(MIN_VALUE).
     * <p>
     */
    public static final double MIN_LOG = Math.log(Double.MIN_VALUE);

    /**
     * Math.log(SQRT2PI);
     */
    public static final double LOGSQRT2PI = Math.log(SQRT2PI);

    // ======================= Error Values
    /**
     * The machine epsilon (macheps) or unit roundoff for <code>double</code> in
     * the Java environment. Machine epsilon gives an upper bound on the
     * relative error due to rounding in floating point arithmetic. Machine
     * epsilon is the smallest number such that (1.0 + EPS != 1.0).
     */
    public static final double EPS = Math.ulp(1.0);

    /**
     * Square-root of the machine epsilon for <code>double</code>.
     */
    public static final double SQRT_EPS = Math.sqrt(EPS);

    /**
     * The machine epsilon (macheps) or unit roundoff for <code>float</code> in
     * the Java environment. Machine epsilon gives an upper bound on the
     * relative error due to rounding in floating point arithmetic. Machine
     * epsilon is the smallest number such that (1F + EPSF != 1F).
     */
    public static final float EPSF = Math.ulp(1F);
    
    /**
     * The smallest positive floating-point number such that 1/xminin is machine
     * representable.
     */
    public final static double XMININ = 2.23e-308;

    /**
     * 150
     */
    public final static int MAX_ITERATIONS = 150;

    /**
     * 4.0 *  Math.ulp(1.0);
     */
    public final static double PRECISION = 4.0 *  Math.ulp(1.0);

}
