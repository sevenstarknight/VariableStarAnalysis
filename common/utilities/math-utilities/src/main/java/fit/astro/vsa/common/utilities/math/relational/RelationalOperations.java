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
package fit.astro.vsa.common.utilities.math.relational;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class RelationalOperations {

    //Empty Constructor
    private RelationalOperations(){
        
    }
    /**
     * Pair down an original set and produce b;
     *
     * @param a
     * @param selectThis
     * @return
     */
    public static RealVector removeRelationOperations(
            RealVector a, List<Boolean> selectThis) {

        List<Double> listOfPoints = new ArrayList<>();
        // Dump into list
        int counter = 0;
        for (Boolean is : selectThis) {

            if (is) {
                // Change Limited Entries in 
                listOfPoints.add(a.getEntry(counter));
            }
            counter++;
        }

        double[] arr = listOfPoints.stream()
                .mapToDouble(Double::doubleValue).toArray();

        // back into real vector
        return MatrixUtils.createRealVector(arr);
    }

    /**
     * Apply the function to a limited set of a and produce b;
     *
     * @param function
     * @param a
     * @param selectThis
     * @return
     */
    public static RealVector applyRelationOperations(UnivariateFunction function,
            RealVector a, List<Boolean> selectThis) {

        // Make Copy
        RealVector b = MatrixUtils.createRealVector(a.toArray());

        int counter = 0;
        for (Boolean is : selectThis) {

            if (is) {
                // Change Limited Entries in 
                double value = function.value(a.getEntry(counter));
                b.setEntry(counter, value);
            }

            counter++;
        }

        return b;
    }

    /**
     * Apply the function to a limited set of a and produce b;
     *
     * @param function
     * @param a
     * @param selectThis
     * @return
     */
    public static List<Double> applyRelationOperations(UnivariateFunction function,
            List<Double> a, List<Boolean> selectThis) {

        // Make Copy
        List<Double> b = new ArrayList<>(a);

        int counter = 0;
        for (Boolean is : selectThis) {
            if (is) {
                // Change Limited Entries in Copy
                b.set(counter, function.value(a.get(counter)));
            }
            counter++;
        }

        return b;
    }

    /**
     *
     * c[idx] = a[idx] oper b
     *
     * @param a
     * @param b
     * @param oper
     * @return
     */
    public static List<Boolean> compareList2Pt(List<Double> a,
            double b, RelationalOperators oper) {

        List<Boolean> c = new ArrayList<>();

        switch (oper) {

            case EQ:
                a.forEach((pt) -> {
                    c.add(Math.abs(pt - b) <= Math.ulp(1.0));
                });
                break;
            case GT:
                a.forEach((pt) -> {
                    c.add(pt > b);
                });
                break;
            case GTEQ:
                a.forEach((pt) -> {
                    c.add(pt >= b);
                });
                break;
            case LT:
                a.forEach((pt) -> {
                    c.add(pt < b);
                });
                break;
            case LTEQ:
                a.forEach((pt) -> {
                    c.add(pt <= b);
                });
                break;
            case NEQ:
                a.forEach((pt) -> {
                    c.add(Math.abs(pt - b) > Math.ulp(1.0));
                });
                break;
            default:
                break;
        }

        return c;
    }

    /**
     *
     * c[idx] = a[idx] oper b
     *
     * @param a
     * @param b
     * @param oper
     * @return
     */
    public static List<Boolean> compareVector2Pt(RealVector a,
            double b, RelationalOperators oper) {

        List<Boolean> c = new ArrayList<>();

        switch (oper) {

            case EQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(Math.abs(a.getEntry(idx) - b) <= Math.ulp(1.0));
                }
                break;
            case GT:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) > b);
                }
                break;
            case GTEQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) >= b);
                }
                break;
            case LT:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) < b);
                }
                break;
            case LTEQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) <= b);
                }
                break;
            case NEQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(Math.abs(a.getEntry(idx) - b) > Math.ulp(1.0));
                }
                break;
            default:
                break;
        }

        return c;
    }

    /**
     *
     * c[idx] = a[idx] oper b[idx]
     *
     * @param a
     * @param b
     * @param oper
     * @return
     */
    public static List<Boolean> compareVector2Vector(RealVector a,
            RealVector b, RelationalOperators oper) {

        if (a.getDimension() != b.getDimension()) {
            throw new ArithmeticException("Dimensions must match");
        }

        List<Boolean> c = new ArrayList<>();

        switch (oper) {

            case EQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(Math.abs(a.getEntry(idx) - b.getEntry(idx)) <= Math.ulp(1.0));
                }
                break;
            case GT:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) > b.getEntry(idx));
                }
                break;
            case GTEQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) >= b.getEntry(idx));
                }
                break;
            case LT:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) < b.getEntry(idx));
                }
                break;
            case LTEQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(a.getEntry(idx) <= b.getEntry(idx));
                }
                break;
            case NEQ:
                for (int idx = 0; idx < a.getDimension(); idx++) {
                    c.add(Math.abs(a.getEntry(idx) - b.getEntry(idx)) > Math.ulp(1.0));
                }
                break;
            default:
                break;
        }

        return c;
    }

    /**
     *
     * c[idx] = a[idx] oper b[idx]
     *
     * @param a
     * @param b
     * @param oper
     * @return
     */
    public static List<Boolean> compareList2List(List<Double> a,
            List<Double> b, RelationalOperators oper) {

        if (a.size() != b.size()) {
            throw new ArithmeticException("Dimensions must match");
        }

        List<Boolean> c = new ArrayList<>();

        switch (oper) {

            case EQ:
                a.forEach((pt) -> {
                    int idx = a.indexOf(pt);
                    c.add(Objects.equals(pt, b.get(idx)));
                });
                break;
            case GT:
                a.forEach((pt) -> {
                    int idx = a.indexOf(pt);
                    c.add(pt > b.get(idx));
                });
                break;
            case GTEQ:
                a.forEach((pt) -> {
                    int idx = a.indexOf(pt);
                    c.add(pt >= b.get(idx));
                });
                break;
            case LT:
                a.forEach((pt) -> {
                    int idx = a.indexOf(pt);
                    c.add(pt < b.get(idx));
                });
                break;
            case LTEQ:
                a.forEach((pt) -> {
                    int idx = a.indexOf(pt);
                    c.add(pt <= b.get(idx));
                });
                break;
            case NEQ:
                a.forEach((pt) -> {
                    int idx = a.indexOf(pt);
                    c.add(!Objects.equals(pt, b.get(idx)));
                });
                break;
            default:
                break;
        }

        return c;
    }

}
