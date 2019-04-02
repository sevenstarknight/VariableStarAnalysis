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
package fit.astro.vsa.common.utilities.io;

import fit.astro.vsa.common.bindings.ml.TimeDomainAttributeMaps;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class ReadingInUCRData {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(ReadingInUCRData.class);

    private final String folderName;

    private final String test = "_TEST";
    private final String train = "_TRAIN";

    public ReadingInUCRData(String folderName) {
        this.folderName = folderName;
    }

    public TimeDomainAttributeMaps getAllData(String fileName) {

        TimeDomainAttributeMaps trainingData = getTrainData(fileName);
        TimeDomainAttributeMaps testingData = getTestData(fileName);

        // This will reference one line at a time
        Map<Integer, Real2DCurve> setOfTimeData = new HashMap<>();
        Map<Integer, String> setOfClasses = new HashMap<>();

        int counter = 0;
        for (Integer idx : trainingData.getSetOfClasses().keySet()) {

            setOfTimeData.put(counter, trainingData.getSetOfWaveforms().get(idx));
            setOfClasses.put(counter, trainingData.getSetOfClasses().get(idx));
            counter++;
        }

        for (Integer idx : testingData.getSetOfClasses().keySet()) {

            setOfTimeData.put(counter, testingData.getSetOfWaveforms().get(idx));
            setOfClasses.put(counter, testingData.getSetOfClasses().get(idx));
            counter++;
        }

        return new TimeDomainAttributeMaps(setOfClasses, setOfTimeData);
    }

    public TimeDomainAttributeMaps getTrainData(String fileName) {

        // The name of the file to open.
        String fileLocation = folderName.concat(File.separator)
                .concat(fileName).concat(File.separator).concat(fileName);
        fileLocation = fileLocation.concat(train);

        return execute(fileLocation);
    }

    public TimeDomainAttributeMaps getTestData(String fileName) {

        // The name of the file to open.
        String fileLocation = folderName.concat(File.separator)
                .concat(fileName).concat(File.separator).concat(fileName);
        fileLocation = fileLocation.concat(test);

        return execute(fileLocation);
    }

    private TimeDomainAttributeMaps execute(String fileLocation) {

        // This will reference one line at a time
        Map<Integer, Real2DCurve> setOfTimeData = new HashMap<>();
        Map<Integer, String> setOfClasses = new HashMap<>();

        try {
            // FileReader reads text files in the default encoding.
            FileReader fileReader
                    = new FileReader(fileLocation);

            // Always wrap FileReader in BufferedReader.
            try (
                    BufferedReader bufferedReader
                    = new BufferedReader(fileReader)) {
                Integer counter = 0;

                String line = null;
                while ((line = bufferedReader.readLine()) != null) {

                    String[] inputData = line.split("\\s+");
                    setOfClasses.put(counter, inputData[1]);

                    String[] inputWaveform
                            = Arrays.copyOfRange(inputData, 2, inputData.length);

                    double[] waveform = new double[inputWaveform.length];

                    for (int idx = 0; idx < inputWaveform.length; idx++) {
                        waveform[idx] = Double.valueOf(inputWaveform[idx]);
                    }

                    RealVector amplitudes = MatrixUtils.createRealVector(waveform);
                    RealVector times = MatrixUtils.createRealVector(VectorOperations.linearSpace(
                            1.0, 0.0, amplitudes.getDimension()));

                    setOfTimeData.put(counter, new Real2DCurve(times, amplitudes));
                    counter++;
                }

                // Always close files.
            }
        } catch (FileNotFoundException ex) {

            LOGGER.error("Unable to open file '" + fileLocation + "'");
        } catch (IOException ex) {
            LOGGER.error("Error reading file '" + fileLocation + "'");
        }

        return new TimeDomainAttributeMaps(setOfClasses, setOfTimeData);

    }

}
