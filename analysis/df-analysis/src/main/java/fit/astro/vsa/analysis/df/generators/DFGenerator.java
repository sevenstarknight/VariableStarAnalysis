/*
 * Copyright (C) 2018 Kyle Johnston 
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
package fit.astro.vsa.analysis.df.generators;

import fit.astro.vsa.analysis.feature.SignalConditioning;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.MathArrays;
import fit.astro.vsa.common.bindings.analysis.MatrixVariateTransform;

/**
 * Haber, R., Rangarajan, A., & Peter, A. M. 2015, in Machine Learning and
 * Knowledge Discovery in Databases (Cham: Springer International Publishing),
 * 20–36
 * <p>
 * Helfer, E., Smith, B. Haber, R., & Peter, A. 2015, Statistical Analysis of
 * Functional Data, Technical Report TR-2015-05„ Tech. rep., Florida Institute
 * of Technology
 *
 * @author Kyle Johnston
 */
public class DFGenerator implements MatrixVariateTransform {

    private final DFOptions dFOptions;

    /**
     *
     * @param dFOptions
     */
    public DFGenerator(DFOptions dFOptions) {
        this.dFOptions = dFOptions;
    }

    @Override
    public int[] getMatrixDimensions() {
        return new int[]{dFOptions.getxDimension(), dFOptions.getyDimension()};
    }

    /**
     *
     * @param phasedWaveform Folded Data
     * @return
     */
    public RealMatrix evaluate(Real2DCurve phasedWaveform) {

        Real2DCurve minMax = SignalConditioning.MinMaxNormalization(phasedWaveform);

        Real2DCurve shiftToMinZero = SignalConditioning.ShiftToMinZero(minMax);

        // ============================================
        // With Convolution
        RealMatrix df = generateTwoDHistogram(shiftToMinZero);

        RealMatrix df_Convolve = convolveMatrixWithKernel(df);

        return df_Convolve;
    }

    /**
     *
     * @param df
     * @return
     */
    private RealMatrix convolveMatrixWithKernel(RealMatrix df) {

        int[] kernelSize = dFOptions.getKernelSize();
        int padding = (kernelSize[0] - 1) / 2;

        RealMatrix convDF1 = MatrixUtils.createRealMatrix(
                dFOptions.getxDimension(),
                dFOptions.getyDimension());

        RealMatrix convDF2 = MatrixUtils.createRealMatrix(
                dFOptions.getxDimension(),
                dFOptions.getyDimension());

        if (dFOptions.getDirection().equals(DFOptions.Directions.both)
                || dFOptions.getDirection().equals(DFOptions.Directions.row)) {

            RealMatrix paddedDF = MatrixUtils.createRealMatrix(
                    dFOptions.getxDimension() + kernelSize[0] - 1,
                    dFOptions.getyDimension());

            paddedDF = paddedDF.scalarAdd(1.0 / dFOptions.getyDimension());

            paddedDF.setSubMatrix(df.getData(), padding, 0);

            //
            RealMatrix convDFColumn = paddedDF.copy();
            for (int idx = 0; idx < dFOptions.getyDimension(); idx++) {

                double[] tmp = MathArrays.convolve(
                        paddedDF.getColumn(idx), dFOptions.getKernel().getRow(0));
                tmp = ArrayUtils.subarray(tmp, padding - 1,
                        tmp.length - padding - 1);

                convDFColumn.setColumn(idx, tmp);
            }
            convDF1 = convDFColumn.getSubMatrix(
                    padding, convDFColumn.getRowDimension() - 1 - padding,
                    0, convDFColumn.getColumnDimension() - 1);
        }

        if (dFOptions.getDirection().equals(DFOptions.Directions.both)
                || dFOptions.getDirection().equals(DFOptions.Directions.column)) {

            RealMatrix paddedDF = MatrixUtils.createRealMatrix(
                    dFOptions.getxDimension(),
                    dFOptions.getyDimension() + kernelSize[0] - 1);

            paddedDF = paddedDF.scalarAdd(1.0 / dFOptions.getxDimension());

            paddedDF.setSubMatrix(df.getData(), 0, padding);

            //
            RealMatrix convDFRow = paddedDF.copy();
            for (int idx = 0; idx < dFOptions.getxDimension(); idx++) {

                double[] tmp = MathArrays.convolve(
                        paddedDF.getRow(idx), dFOptions.getKernel().getRow(0));
                tmp = ArrayUtils.subarray(tmp, padding - 1,
                        tmp.length - padding - 1);

                convDFRow.setRow(idx, tmp);
            }
            convDF2 = convDFRow.getSubMatrix(
                    0, convDFRow.getRowDimension() - 1,
                    padding, convDFRow.getColumnDimension() - 1 - padding);
        }

        // Enforce DF Field
        RealMatrix tmpConv = convDF1.add(convDF2);
        RealMatrix convDF = new Array2DRowRealMatrix(dFOptions.getxDimension(),
                dFOptions.getyDimension());

        for (int idx = 0; idx < tmpConv.getColumnDimension(); idx++) {
            double total = tmpConv.getColumnVector(idx).getL1Norm();
            convDF.setColumnVector(idx, tmpConv.getColumnVector(idx).mapDivide(total));
        }

        return convDF;
    }

    /**
     *
     * @param phasedWaveform
     * @return
     */
    private RealMatrix generateTwoDHistogram(Real2DCurve phasedWaveform) {

        RealMatrix frequencyDist = MatrixUtils.createRealMatrix(
                dFOptions.getxDimension(), dFOptions.getyDimension());

        for (int kdx = 0; kdx < phasedWaveform.size(); kdx++) {

            double xValue = phasedWaveform.xValueAt(kdx);
            double yValue = phasedWaveform.yValueAt(kdx);

            List<Double> xList = new ArrayList<>(dFOptions.getxStates());

            int xIdx = 0;
            for (int idx = 0; idx < xList.size() - 1; idx++) {
                if (xList.get(idx) < xValue && xList.get(idx + 1) >= xValue) {
                    xIdx = idx;
                    break;
                }
            }

            List<Double> yList = new ArrayList<>(dFOptions.getyStates());
            int yIdx = 0;
            for (int idx = 0; idx < yList.size() - 1; idx++) {
                if (yList.get(idx) < yValue && yList.get(idx + 1) >= yValue) {
                    yIdx = idx;
                    break;
                }
            }

            frequencyDist.addToEntry(xIdx, yIdx, 1.0);
        }

        RealVector sumColumns = MatrixOperations
                .dimensionalSummation(frequencyDist, Boolean.TRUE);

        RealMatrix df = MatrixUtils.createRealMatrix(
                dFOptions.getxDimension(), dFOptions.getyDimension());
        for (int idx = 0; idx < dFOptions.getyDimension(); idx++) {

            RealVector column = frequencyDist.getColumnVector(idx);

            RealVector tmp = column.mapDivide(sumColumns.getEntry(idx));

            if (tmp.isNaN()) {
                for (int jdx = 0; jdx < tmp.getDimension(); jdx++) {
                    if (Double.isNaN(tmp.getEntry(jdx))) {
                        tmp.setEntry(jdx, 0.0);
                    }
                }
            }

            df.setColumnVector(idx, tmp);
        }

        return df;
    }

}
