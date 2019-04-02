/*
*   MathTools  -- A collection of useful math utility routines.
*   <p>
*   Copyright (C) 1999-2014 by Joseph A. Huwaldt
*   All rights reserved.
*   <p>
*   This library is free software; you can redistribute it and/or
*   modify it under the terms of the GNU Lesser General Public
*   License as published by the Free Software Foundation; either
*   version 2 of the License, or (at your option) any later version.
*   <p>
*   This library is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without isEven the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*   Lesser General Public License for more details.
*   <p>
*   You should have received a copy of the GNU Lesser General Public License
*   along with this program; if not, write to the Free Software
*   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*   Or visit:  http://www.gnu.org/licenses/lgpl.html
**/
package fit.astro.vsa.common.utilities.math;

/**
 * Collection of Numerical Tests Borrowed from:
 * <p>
 * http://thehuwaldtfamily.org/java/index.html
 * <p>
 * @author Joseph A. Huwaldt Date: September 29, 1997
 * @version June 1, 2015
 * <p>
 */
public class NumericTests {

    //Empty Constructor
    private NumericTests() {
    }

    //-----------------------------------------------------------------------------------
    /**
     * Test to see if a given long integer is even.
     * <p>
     * @param n Integer number to be tested.
     * <p>
     * @return True if the number is isEven, false if it is isOdd.
     * <p>
     */
    public static boolean isEven(long n) {
        return (n & 1) == 0;
    }

    /**
     * Test to see if a given long integer is odd.
     * <p>
     * @param n Integer number to be tested.
     * <p>
     * @return True if the number is isOdd, false if it is isEven.
     * <p>
     */
    public static boolean isOdd(long n) {
        return (n & 1) != 0;
    }

    /**
     * Returns the absolute value of "a" times the sign of "b".
     * <p>
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static double sign(double a, double b) {
        return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
    }

    /**
     * Returns the absolute value of "a" times the sign of "b".
     * <p>
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static float sign(float a, double b) {
        return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
    }

    /**
     * Returns the absolute value of "a" times the sign of "b".
     * <p>
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static long sign(long a, double b) {
        return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
    }

    /**
     * Returns the absolute value of "a" times the sign of "b".
     * <p>
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static int sign(int a, double b) {
        return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
    }

    //=============================================================
    /**
     * Returns true if the two supplied numbers are approximately equal to
     * within machine precision.
     * <p>
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static boolean isApproxEqual(double a, double b) {
        double eps2 = Math.ulp(a);
        double eps = (eps2 > Math.ulp(1.0) ? eps2 : Math.ulp(1.0));
        return Math.abs(a - b) <= eps;
    }

    /**
     * Returns true if the two supplied numbers are approximately equal to
     * within the specified tolerance.
     * <p>
     * @param a
     * @param b
     * @param tol
     * <p>
     * @return
     */
    public static boolean isApproxEqual(double a, double b, double tol) {
        return Math.abs(a - b) <= tol;
    }

    /**
     * Returns true if the two supplied numbers are approximately equal to
     * within machine precision.
     * <p>
     * @param a
     * @param b
     * <p>
     * @return
     */
    public static boolean isApproxEqual(float a, float b) {
        float eps2 = Math.ulp(a);
        float eps = (eps2 > Math.ulp(1F) ? eps2 : Math.ulp(1F));
        return Math.abs(a - b) <= eps;
    }

    /**
     * Returns true if the two supplied numbers are approximately equal to
     * within the specified tolerance.
     * <p>
     * @param a
     * @param b
     * @param tol
     * <p>
     * @return
     */
    public static boolean isApproxEqual(float a, float b, float tol) {
        return Math.abs(a - b) <= tol;
    }

    /**
     * Returns true if the supplied number is approximately zero to within
     * machine precision.
     * <p>
     * @param a
     * <p>
     * @return
     */
    public static boolean isApproxZero(double a) {
        return Math.abs(a) <= Math.ulp(1.0);
    }

    /**
     * Returns true if the supplied number is approximately zero to within the
     * specified tolerance.
     * <p>
     * @param a
     * @param tol
     * <p>
     * @return
     */
    public static boolean isApproxZero(double a, double tol) {
        return Math.abs(a) <= tol;
    }

    /**
     * Returns true if the supplied number is approximately zero to within
     * machine precision.
     * <p>
     * @param a
     * <p>
     * @return
     */
    public static boolean isApproxZero(float a) {
        return Math.abs(a) <= Math.ulp(1F);
    }

    /**
     * Returns true if the supplied number is approximately zero to within the
     * specified tolerance.
     * <p>
     * @param a
     * @param tol
     * <p>
     * @return
     */
    public static boolean isApproxZero(float a, float tol) {
        return Math.abs(a) <= tol;
    }
}
