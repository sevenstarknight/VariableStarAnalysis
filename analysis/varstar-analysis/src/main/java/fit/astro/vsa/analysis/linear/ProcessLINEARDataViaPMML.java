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
package fit.astro.vsa.analysis.linear;

import fit.astro.vsa.analysis.ProcessTimeDomainViaPMML;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData.Views;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.analysis.raw.SupportFunctionality;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ProcessLINEARDataViaPMML extends ProcessTimeDomainViaPMML {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessLINEARDataViaPMML.class);

    private static final Random RAND = new Random(42L);

    private static VarStarDataset dataset;

    private static CanonicalVariates canonicalVariates_SSMM;
    private static CanonicalVariates canonicalVariates_DF;

    private static Pair<Map<String, RealVector>, Map<String, RealVector>> transformationVectors;

    public static void main(String[] args) {

        String fileLocation = "LinearData/LinearData.mat";

        // =================================================================
        double tau = 0.25;
        if (args.length > 0) {
            try {
                tau = Double.parseDouble(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be a double. Using 0.25 instead");
                tau = 0.25;
            }
        }

        double mu = 5.0;
        if (args.length > 1) {
            try {
                mu = Double.parseDouble(args[1]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[1] + " must be a double. Using 5.0 instead");
                mu = 5.0;
            }
        }

        double lambda = 0.5;
        if (args.length > 2) {
            try {
                lambda = Double.parseDouble(args[2]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[2] + " must be a double. Using 0.5 instead");
                lambda = 0.5;
            }
        }

        boolean isTest = Boolean.FALSE;
        if (args.length > 3) {
            String testOption = args[3];
            if (testOption.equalsIgnoreCase("-test")) {
                isTest = Boolean.TRUE;
            }
        }

        try {
            execute(tau, mu, lambda, isTest, fileLocation);
        } catch (IOException | NotEnoughDataException ex) {
            LOGGER.warn(ex.getMessage());
        }
    }

    /**
     * Run testing procedure
     *
     * @param tau
     * @param mu
     * @param lambda
     * @param isTest
     * @param fileLocation
     * @throws IOException
     * @throws NotEnoughDataException
     */
    public static void execute(double tau, double mu, double lambda, boolean isTest, String fileLocation)
            throws IOException, NotEnoughDataException {

        ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

        dataset = lINEARData.getLinearDataset();

        generatePatterns();
        // =================================================================
        if (!isTest) {
            trainPMML(tau, mu, lambda, "Linear");
        } else {
            testPMML(13, tau, mu, lambda, "Linear");
        }
    }

    /**
     *
     * @param phasedData
     * @param setOfClasses
     */
    private static void generatePatterns() {

        Map<Integer, String> setOfClasses = dataset.getMultiViewSideData().getSetOfClasses();

        Map<Integer, Real2DCurve> phasedData = SupportFunctionality.generatePhasedData(dataset);

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

        Map<Integer, RealVector> setOfPatterns_SSMM = new HashMap<>();
        Map<Integer, RealVector> setOfPatterns_DF = new HashMap<>();

        TrainCrossTestGenerator trainTest
                = new TrainCrossTestGenerator(setOfClasses, 0.5, RAND);

        crossvalMap = trainTest.getCrossvalMap();

        for (Integer idx : setOfClasses.keySet()) {

            Real2DCurve phasedWaveform = phasedData.get(idx);
            Real2DCurve currentWaveform = dataset.getTimeD().get(idx);

            // input waveform is already phased for UCR data
            RealMatrix df = dfGenerator.evaluate(phasedWaveform);

            setOfPatterns_DF.put(idx, new ArrayRealVector(
                    MatrixOperations.unpackMatrix(df)));

            // input waveform doesn't require phasing
            RealMatrix ssmm = ssmmGenerator.evaluate(currentWaveform);

            setOfPatterns_SSMM.put(idx, new ArrayRealVector(
                    MatrixOperations.unpackMatrix(ssmm)));

            setOfClasses.put(idx, setOfClasses.get(idx));

        }

        // ======================================================
        // Generate Training Data
        List<Integer> keysTraining = trainTest.getTrainingData();
        
        setOfClasses_Training = keysTraining.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));
        // ============================================================

        ECVA ecva_SSMM = new ECVA(keysTraining.stream()
                .filter(setOfPatterns_SSMM::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfPatterns_SSMM::get)),
                setOfClasses_Training);

        canonicalVariates_SSMM = ecva_SSMM.execute();

        ECVA ecva_DF = new ECVA(keysTraining.stream()
                .filter(setOfPatterns_DF::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfPatterns_DF::get)),
                setOfClasses_Training);

        canonicalVariates_DF = ecva_DF.execute();

        // ======================================================
        // Featrure Space Multi-View
        Map<Integer, Map<String, RealVector>> setOfPatterns_Training_tmp = new HashMap<>();

        for (Integer idx : keysTraining) {

            setOfPatterns_Training_tmp.put(idx, new HashMap<>());
            setOfPatterns_Training_tmp.get(idx).put(Views.SSMM.toString(), canonicalVariates_SSMM.getCanonicalVariates().get(idx));
            setOfPatterns_Training_tmp.get(idx).put(Views.DF.toString(), canonicalVariates_DF.getCanonicalVariates().get(idx));
            
            setOfPatterns_Training_tmp.get(idx).put(Views.Photometric.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.Photometric.toString()));

            setOfPatterns_Training_tmp.get(idx).put(Views.TimeDomainVar.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.TimeDomainVar.toString()));

        }

        transformationVectors = NormalizeData.normalizeMultiViewVectorVariate(setOfPatterns_Training_tmp);

        setOfPatterns_Training = NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Training_tmp,
                transformationVectors);

        // ======================================================
        List<Integer> keysTesting = trainTest.getTestingData();

        Map<Integer, Map<String, RealVector>> setOfPatterns_Testing_tmp = new HashMap<>();

        setOfClasses_Testing = keysTesting.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        for (Integer idx : keysTesting) {

            setOfPatterns_Testing_tmp.put(idx, new HashMap<>());
            setOfPatterns_Testing_tmp.get(idx).put(Views.SSMM.toString(), canonicalVariates_SSMM.applyECVA(setOfPatterns_SSMM.get(idx)));
            setOfPatterns_Testing_tmp.get(idx).put(Views.DF.toString(), canonicalVariates_DF.applyECVA(setOfPatterns_DF.get(idx)));
            
            setOfPatterns_Testing_tmp.get(idx).put(Views.Photometric.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.Photometric.toString()));

            setOfPatterns_Testing_tmp.get(idx).put(Views.TimeDomainVar.toString(), dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.TimeDomainVar.toString()));

        }

        setOfPatterns_Testing = NormalizeData.applyNormalizeVectorVariate(setOfPatterns_Testing_tmp,
                transformationVectors);

    }

}
