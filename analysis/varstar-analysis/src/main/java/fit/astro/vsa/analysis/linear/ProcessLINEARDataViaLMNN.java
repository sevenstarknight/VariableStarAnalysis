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

import fit.astro.vsa.analysis.ConcatonatedData;
import fit.astro.vsa.analysis.ProcessRawLMNN;
import fit.astro.vsa.analysis.df.generators.DFGenerator;
import fit.astro.vsa.analysis.df.generators.DFOptions;
import fit.astro.vsa.analysis.ssmm.generators.SSMMGenerator;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.utilities.ml.ecva.CanonicalVariates;
import fit.astro.vsa.utilities.ml.ecva.ECVA;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData.Views;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.analysis.raw.SupportFunctionality;
import fit.astro.vsa.analysis.ucr.ProcessUCRDataViaNCA;
import fit.astro.vsa.utilities.ml.training.NormalizeData;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import fit.astro.vsa.common.utilities.io.SerialStorage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
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
public class ProcessLINEARDataViaLMNN extends ProcessRawLMNN {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ProcessLINEARDataViaLMNN.class);

    private static final Random RAND = new Random(42L);

    private static VarStarDataset dataset;

    private static CanonicalVariates canonicalVariates_SSMM;
    private static CanonicalVariates canonicalVariates_DF;

    private static Pair<RealVector, RealVector> transformationVectors;

    private static final String resourcePath = "src"
            + File.separator + "main"
            + File.separator + "resources";

    private static final String dataLocations = "/datasets/dataLINEAR.ser";

    public static void main(String[] args) {

        String fileLocation = "LinearData/LinearData.mat";

        // =================================================================
        int kValue = 3;
        if (args.length > 0) {
            try {
                kValue = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("Argument" + args[0] + " must be a int. Using 3 instead");
                kValue = 3;
            }
        }

        boolean isTest = Boolean.FALSE;
        if (args.length > 1) {
            String testOption = args[1];
            if (testOption.equalsIgnoreCase("-test")) {
                isTest = Boolean.TRUE;
            }
        }

        try {
            execute(isTest, fileLocation);
        } catch (IOException | NotEnoughDataException ex) {
            LOGGER.warn(ex.getMessage());
        }
    }

    /**
     * Run testing procedure
     *
     * @param isTest
     * @param fileLocation
     * @throws IOException
     * @throws NotEnoughDataException
     */
    public static void execute(boolean isTest, String fileLocation)
            throws IOException, NotEnoughDataException {

        ObjectInputStream oisData;
        try (InputStream finData = ProcessLINEARDataViaLMNN.class
                .getResourceAsStream(dataLocations)) {
            oisData = new ObjectInputStream(finData);
            ConcatonatedData dataIN = (ConcatonatedData) oisData.readObject();
            finData.close();
            oisData.close();

            setOfPatterns_Training = dataIN.getSetOfPatterns_Training();
            setOfPatterns_Testing = dataIN.getSetOfPatterns_Testing();

            setOfClasses_Training = dataIN.getSetOfClasses_Training();
            setOfClasses_Testing = dataIN.getSetOfClasses_Testing();

            crossvalMap = dataIN.getCrossvalMap();

        } catch (ClassNotFoundException | NullPointerException exception) {
            LOGGER.error(null, exception);
            // Pull Data
            ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

            dataset = lINEARData.getLinearDataset();

            generatePatterns();
        }
        // Pull Data
        // =================================================================
        if (!isTest) {
            trainKNN();
        } else {
            testKNN(3);
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
        Map<Integer, RealVector> setOfPatterns_Training_tmp = new HashMap<>();

        for (Integer idx : keysTraining) {

            RealVector tmp = (canonicalVariates_SSMM.getCanonicalVariates().get(idx));
            tmp = tmp.append(canonicalVariates_DF.getCanonicalVariates().get(idx));
            tmp = tmp.append(dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.Photometric.toString()));
            tmp = tmp.append(dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.TimeDomainVar.toString()));

            setOfPatterns_Training_tmp.put(idx, tmp);

        }

        transformationVectors = NormalizeData.normalizeVector(setOfPatterns_Training_tmp);

        setOfPatterns_Training = NormalizeData.applyNormVector(setOfPatterns_Training_tmp,
                transformationVectors);

        // ======================================================
        List<Integer> keysTesting = trainTest.getTestingData();

        Map<Integer, RealVector> setOfPatterns_Testing_tmp = new HashMap<>();

        setOfClasses_Testing = keysTesting.stream().filter(setOfClasses::containsKey)
                .collect(Collectors.toMap(Function.identity(), setOfClasses::get));

        for (Integer idx : keysTesting) {

            RealVector tmp = (canonicalVariates_SSMM.applyECVA(setOfPatterns_SSMM.get(idx)));
            tmp = tmp.append(canonicalVariates_DF.applyECVA(setOfPatterns_DF.get(idx)));

            tmp = tmp.append(dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.Photometric.toString()));
            tmp = tmp.append(dataset.getMultiViewSideData()
                    .getMapOfVectorPatterns().get(idx).get(Views.TimeDomainVar.toString()));

            setOfPatterns_Testing_tmp.put(idx, tmp);

        }

        setOfPatterns_Testing = NormalizeData.applyNormVector(setOfPatterns_Testing_tmp,
                transformationVectors);

        ConcatonatedData concatonatedData = new ConcatonatedData(setOfPatterns_Training,
                setOfClasses_Training, setOfPatterns_Testing, setOfClasses_Testing, crossvalMap);

        String path = resourcePath + dataLocations;

        try {
            SerialStorage.storeSerialObject(concatonatedData, new File(path));
        } catch (FileNotFoundException exception) {
            LOGGER.error(null, exception);
        }
    }

}
