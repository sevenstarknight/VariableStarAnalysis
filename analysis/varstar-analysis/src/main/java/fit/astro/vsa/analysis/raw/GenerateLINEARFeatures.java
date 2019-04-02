/*
 * Copyright (C) 2019 kjohnston
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
package fit.astro.vsa.analysis.raw;

import fit.astro.vsa.analysis.linear.RawKnn_LINEAR_Vector;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.analysis.feature.smoothing.SuperSmoother;
import fit.astro.vsa.analysis.feature.smoothing.util.SuperSmootherProperties;
import fit.astro.vsa.common.bindings.math.vector.ModFunction;
import java.io.IOException;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class GenerateLINEARFeatures {

    private static final Random RAND = new Random(42L);

    private static final Logger LOGGER
            = LoggerFactory.getLogger(RawKnn_LINEAR_Vector.class);

    private static Map<Integer, Map<String, RealVector>> setOfPatterns;
    private static Map<Integer, String> setOfClasses;

    private static VarStarDataset dataset;

    public static void main(String[] args) throws IOException, NotEnoughDataException {

        String fileLocation = "/Users/kjohnston/Google Drive/VarStarData/LinearData/LinearData.mat";

        ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

        dataset = lINEARData.getLinearDataset();

        LOGGER.info(LocalDateTime.now().toString());
        LocalDateTime start = LocalDateTime.now();

        Map<Integer, Real2DCurve> phasedSmoothData = generateSmoothPhasedData(dataset);
        Map<Integer, Real2DCurve> phasedData = generatePhasedData(dataset);

        generatePatterns(phasedSmoothData);

        // ========
        List<MLArray> list = new ArrayList<>();

        MLStructure tmpStruct = new MLStructure("Phased Data", new int[]{1, phasedData.size()});

        for (Integer idx : phasedData.keySet()) {
            double[][] time = new double[][]{(phasedData.get(idx)).getXArrayPrimitive()};
            double[][] amp = new double[][]{(phasedData.get(idx)).getYArrayPrimitive()};

            tmpStruct.setField("time", new MLDouble("time", time), idx);
            tmpStruct.setField("amp", new MLDouble("amp", amp), idx);
        }

        list.add(tmpStruct);

        // ========

        MatlabFunctions.storeToFinal("LINEAR-Features", list);

        LOGGER.info(LocalDateTime.now().toString());
        LocalDateTime stop = LocalDateTime.now();

        double timeToProcess = ((double) Duration.between(start, stop).getSeconds());
        LOGGER.info("Time to process: " + timeToProcess);
    }

    /**
     *
     * @param phasedData
     * @param setOfClasses
     */
    private static void generatePatterns(Map<Integer, Real2DCurve> phasedData) {

        setOfClasses = dataset.getMultiViewSideData().getSetOfClasses();

        // =======================================================
        // Feature Processing
        DFOptions dFOptions = new DFOptions(30, 25,
                new int[]{7, 1}, 0.4, DFOptions.Directions.both);

        DFGenerator dfGenerator
                = new DFGenerator(dFOptions);

        // =======================================================
        double windowWidth = SSMMGenerator.estimateWindowWidth(
                phasedData.values().iterator().next(), 35);

        SSMMGenerator ssmmGenerator
                = new SSMMGenerator(0.06, windowWidth);

        setOfPatterns = new HashMap<>(setOfClasses.keySet().size());

        for (Integer idx : setOfClasses.keySet()) {

            Map<String, RealVector> pattern = new HashMap<>();

            Real2DCurve phasedWaveform = phasedData.get(idx);
            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            double period = dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(
                    ReadingInLINEARData.Views.TimeDomainVar.toString()).getEntry(0);

            DescriptiveStatistics descript = new DescriptiveStatistics(currentWaveform
                    .getYArrayPrimitive());

            RealVector statistics = new ArrayRealVector(
                    new double[]{period, descript.getMean(), descript.getStandardDeviation(),
                        descript.getSkewness(), descript.getKurtosis()});

            pattern.put(ReadingInLINEARData.Views.Statistics.toString(), statistics);

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(phasedWaveform);

            pattern.put(ReadingInLINEARData.Views.DF.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            pattern.put(ReadingInLINEARData.Views.SSMM.toString(), new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));

            setOfPatterns.put(idx, pattern);
            setOfClasses.put(idx, setOfClasses.get(idx));

        }

    }

        /**
     *
     * @param dataset
     * @return
     */
    private static Map<Integer, Real2DCurve> generatePhasedData(VarStarDataset dataset) {


        Map<Integer, Real2DCurve> phasedData = new HashMap<>();

        for (Integer idx : dataset.getTimeD().keySet()) {

            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            // ============================================================
            // Phase the data
            double truePeriod = Math.pow(10, dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.TimeDomainVar.toString()).getEntry(0));

            RealVector time = currentWaveform.getXVector();

            RealVector phasedVector = (time.mapDivide(truePeriod)).map(new ModFunction(1.0));

            Real2DCurve phased2D = new Real2DCurve(phasedVector, currentWaveform.getYVector());

            // ============================================================
            // Sorted Phased Data With Window
            Real2DCurve sortedSeries = phased2D.getSortedSeries();

            Real2DCurve sortedSeriesMinus = new Real2DCurve(sortedSeries.getXVector().mapSubtract(1.0),
                    sortedSeries.getYVector());

            Real2DCurve sortedSeriesPlus = new Real2DCurve(sortedSeries.getXVector().mapAdd(1.0),
                    sortedSeries.getYVector());

            Real2DCurve sortedPlusMinus = new Real2DCurve(
                    sortedSeriesMinus.getXVector().append(sortedSeries.getXVector()).append(sortedSeriesPlus.getXVector()),
                    sortedSeriesMinus.getYVector().append(sortedSeries.getYVector()).append(sortedSeriesPlus.getYVector())
            );


            phasedData.put(idx, sortedPlusMinus);
        }

        return phasedData;
    }

    
    /**
     *
     * @param dataset
     * @return
     */
    private static Map<Integer, Real2DCurve> generateSmoothPhasedData(VarStarDataset dataset) {

        SuperSmootherProperties props
                = new SuperSmootherProperties();

        Map<Integer, Real2DCurve> phasedData = new HashMap<>();

        for (Integer idx : dataset.getTimeD().keySet()) {

            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            // ============================================================
            // Phase the data
            double truePeriod = Math.pow(10, dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(ReadingInLINEARData.Views.TimeDomainVar.toString()).getEntry(0));

            RealVector time = currentWaveform.getXVector();

            RealVector phasedVector = (time.mapDivide(truePeriod)).map(new ModFunction(1.0));

            Real2DCurve phased2D = new Real2DCurve(phasedVector, currentWaveform.getYVector());

            // ============================================================
            // Sorted Phased Data With Window
            Real2DCurve sortedSeries = phased2D.getSortedSeries();

            Real2DCurve sortedSeriesMinus = new Real2DCurve(sortedSeries.getXVector().mapSubtract(1.0),
                    sortedSeries.getYVector());

            Real2DCurve sortedSeriesPlus = new Real2DCurve(sortedSeries.getXVector().mapAdd(1.0),
                    sortedSeries.getYVector());

            Real2DCurve sortedPlusMinus = new Real2DCurve(
                    sortedSeriesMinus.getXVector().append(sortedSeries.getXVector()).append(sortedSeriesPlus.getXVector()),
                    sortedSeriesMinus.getYVector().append(sortedSeries.getYVector()).append(sortedSeriesPlus.getYVector())
            );

            // ============================================================
            // Smoother
            SuperSmoother superSmoother = new SuperSmoother(sortedPlusMinus, props);

            RealVector ySmooth = superSmoother.execute().getSmo_n()
                    .getSubVector(sortedSeriesMinus.size(), sortedSeriesMinus.size());

            phasedData.put(idx, new Real2DCurve(sortedSeries.getXVector(), ySmooth));
        }

        return phasedData;
    }

}
