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
package fit.astro.vsa.common.bindings.math;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * <p>
 * @author Kyle Johnston
 */
public class Real2DCurve implements Serializable {

    private static final long serialVersionUID = -6107478754148091667L;

    // <editor-fold defaultstate="collapsed" desc="Variables">

    private final RealVector ySeries;
    private final RealVector xSeries;
    private final RealVector wSeries;

    private final String NOTSAMELENGTH = "Array Series not of same length";

    // </editor-fold>
    // Constructor method. Initializes the ArrayList.
    /**
     * Empty 2D Series (for initialization)
     * <p>
     * @param size
     */
    public Real2DCurve(int size) {
        this.ySeries = new ArrayRealVector(size);
        this.xSeries = new ArrayRealVector(size);
        this.wSeries = new ArrayRealVector(size);
    }

    /**
     * Empty 2D Series (for initialization)
     * <p>
     * @param weightedPoints
     */
    public Real2DCurve(List<WeightedObservedPoint> weightedPoints) {
        this.ySeries = new ArrayRealVector(weightedPoints.size());
        this.xSeries = new ArrayRealVector(weightedPoints.size());
        this.wSeries = new ArrayRealVector(weightedPoints.size());

        int idx = 0;
        for (WeightedObservedPoint wpt : weightedPoints) {
            ySeries.setEntry(idx, wpt.getY());
            xSeries.setEntry(idx, wpt.getX());
            wSeries.setEntry(idx, wpt.getWeight());
            idx++;
        }

    }

    /**
     * Make a copy
     *
     * @param twoDSeries
     */
    public Real2DCurve(Real2DCurve twoDSeries) {
        this.ySeries = twoDSeries.getYVector();
        this.xSeries = twoDSeries.getXVector();
        this.wSeries = twoDSeries.getWVector();

    }

    /**
     * Fill 2d series with double[]
     * <p>
     * @param xSeries
     * @param ySeries
     */
    public Real2DCurve(double[] xSeries, double[] ySeries) {

        if (xSeries.length != ySeries.length) {
            throw new ArithmeticException(NOTSAMELENGTH);
        }

        this.ySeries = new ArrayRealVector(ySeries);
        this.xSeries = new ArrayRealVector(xSeries);

        double[] onesArray = new double[xSeries.length];
        Arrays.fill(onesArray, 1.0);
        this.wSeries = new ArrayRealVector(onesArray);

    }

    /**
     * Fill 2d series with Real Vector
     * <p>
     * @param xSeries
     * @param ySeries
     */
    public Real2DCurve(RealVector xSeries, RealVector ySeries) {
        if (xSeries.getDimension() != ySeries.getDimension()) {
            throw new ArithmeticException(NOTSAMELENGTH);
        }
        this.ySeries = ySeries;
        this.xSeries = xSeries;

        double[] onesArray = new double[xSeries.getDimension()];
        Arrays.fill(onesArray, 1.0);
        this.wSeries = new ArrayRealVector(onesArray);

    }

    /**
     * Fill 2d series with Real Vector
     * <p>
     * @param xSeries
     * @param ySeries
     * @param wSeries
     */
    public Real2DCurve(RealVector xSeries, RealVector ySeries, RealVector wSeries) {
        if (xSeries.getDimension() != ySeries.getDimension()) {
            throw new ArithmeticException(NOTSAMELENGTH);
        }

        if (wSeries.getDimension() != ySeries.getDimension()) {
            throw new ArithmeticException(NOTSAMELENGTH);
        }

        this.ySeries = ySeries;
        this.xSeries = xSeries;
        this.wSeries = wSeries;

    }

    /**
     * Appends a point to the curve defined by its 2-dimensional coordinates.
     * Weight added = 1;
     * <p>
     * @param y
     * @param x double x-coordinate of the point
     */
    public void addPair(double y, double x) {
        ySeries.append(y);
        xSeries.append(x);
        wSeries.append(1.0);
    }

    /**
     * Change the value of a point on the curve defined by its index. Weight
     * added = 1;
     * <p>
     * @param i
     * @param y
     * @param x
     */
    public void changePairAt(int i, double y, double x) {
        ySeries.addToEntry(i, y);
        xSeries.addToEntry(i, x);
        wSeries.addToEntry(i, 1.0);
    }

    /**
     * Change a point on the curve defined by its index
     * <p>
     * @param i
     * @param y
     * @param x
     * @param w
     */
    public void changeWeightedPairAt(int i, double y, double x, double w) {
        ySeries.addToEntry(i, y);
        xSeries.addToEntry(i, x);
        wSeries.addToEntry(i, w);

    }

    /**
     * Appends a point to the curve
     * <p>
     * @param y
     * @param x
     * @param w weight
     */
    public void addWeightedPair(double y, double x, double w) {
        ySeries.append(y);
        xSeries.append(x);
        wSeries.append(w);
    }

    /**
     *
     * @param i the index
     * <p>
     * @return [x,y] the pair at i
     */
    public RealVector getPairAt(int i) {
        return new ArrayRealVector(
                new double[]{xSeries.getEntry(i), ySeries.getEntry(i)});
    }

    /**
     *
     * @param idx The index
     * <p>
     * @return [w,x,y] the weighted point
     */
    public WeightedObservedPoint getWeightedPointAt(int idx) {
        return new WeightedObservedPoint(
                wSeries.getEntry(idx),
                xSeries.getEntry(idx),
                ySeries.getEntry(idx));
    }

    /**
     * Set the pair value at the given input index
     *
     * @param i [index]
     * @param vector [x,y]
     */
    public void setPairAt(int i, RealVector vector) {
        xSeries.setEntry(i, vector.getEntry(0));
        ySeries.setEntry(i, vector.getEntry(1));
    }

    /**
     *
     * @return The set of weighted points
     */
    public List<WeightedObservedPoint> getListOfWeightedPoints() {
        List<WeightedObservedPoint> pts = new ArrayList<>(xSeries.getDimension());
        for (int idx = 0; idx < xSeries.getDimension(); idx++) {
            pts.add(getWeightedPointAt(idx));
        }

        return pts;
    }

    /**
     * Return the 2D series sorted based on x component, does not sort the data
     * in the object, it only provides the sorted output
     *
     * @return The sorted series of points from 2D
     */
    public Real2DCurve getSortedSeries() {
        List<WeightedObservedPoint> pts = getListOfWeightedPoints();
        Collections.sort(pts, new SortByX());

        return new Real2DCurve(pts);
    }

    /**
     * Get the double-double array version of the data
     *
     * @return
     */
    public double[][] getArray() {
        RealMatrix ts = new Array2DRowRealMatrix(size(), 2);
        ts.setColumnVector(0, xSeries);
        ts.setColumnVector(1, ySeries);

        return ts.getData();
    }

    // <editor-fold defaultstate="collapsed" desc="Setters and Getters">
    /**
     * @return int the number of points in the curve.
     */
    public int size() {
        return xSeries.getDimension();
    }

    /**
     * @return double the y coordinate of the point at the given index.
     * <p>
     * @param index the index of the point.
     */
    public double yValueAt(int index) {
        return ySeries.getEntry(index);
    }

    /**
     * @return double the x coordinate of the point at the given index.
     * <p>
     * @param index the index of the point.
     */
    public double xValueAt(int index) {
        return xSeries.getEntry(index);
    }

    /**
     *
     * @return
     */
    public double[] getYArrayPrimitive() {
        return ySeries.toArray();
    }

    /**
     *
     * @return
     */
    public double[] getXArrayPrimitive() {
        return xSeries.toArray();
    }

    /**
     *
     * @return
     */
    public Double[] getYArray() {
        return ArrayUtils.toObject(ySeries.toArray());
    }

    /**
     *
     * @return
     */
    public RealVector getYVector() {
        return ySeries;
    }

    /**
     *
     * @return
     */
    public Double[] getXArray() {
        return ArrayUtils.toObject(xSeries.toArray());
    }

    /**
     *
     * @return
     */
    public RealVector getXVector() {
        return xSeries;
    }

    /**
     * @return the wSeries
     */
    public RealVector getWVector() {
        return wSeries;
    }

    /**
     *
     * @return
     */
    public Double[] getWArray() {
        return ArrayUtils.toObject(xSeries.toArray());
    }

    /**
     *
     * @return
     */
    public double[] getWArrayPrimitive() {
        return xSeries.toArray();
    }

    // </editor-fold>
    private static class SortByX implements Comparator<WeightedObservedPoint> {

        public SortByX() {

        }

        @Override
        public int compare(WeightedObservedPoint o1, WeightedObservedPoint o2) {
            double delta = o1.getX() - o2.getX();

            if (delta > 0) {
                return 1;
            } else if (delta < 0) {
                return -1;
            } else {
                return 0;
            }

        }
    }
}
