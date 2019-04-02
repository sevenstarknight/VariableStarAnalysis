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

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MatlabFunctions {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(MatlabFunctions.class);

    // Empty Constructor
    private MatlabFunctions() {
    }

    /**
     * Create a Matlab file reader.
     *
     * @param inStreamSignal
     * @return
     * @throws IOException
     */
    public static MatFileReader generateMatFileReader(
            InputStream inStreamSignal)
            throws IOException {
        //==============================================================
        // temp fix to accomdate JMatIO
        File tempFile = File.createTempFile("mat2File", ".tmp");
        tempFile.deleteOnExit();
        MatFileReader readIn;
        try (FileOutputStream out = new FileOutputStream(tempFile)) {
            IOUtils.copy(inStreamSignal, out);

            readIn = new MatFileReader(tempFile);

            out.flush();
            out.close();
        }

        //===============================================================
        return readIn;
    }

    /**
     * Write data to file.
     *
     * @param file
     * @param list
     */
    public static void storeToFile(File file, List<MLArray> list) {
        // =============================================================
        try {
            // Cycle
            if (file.exists()) {
                file.delete();
            }
            file.createNewFile();

            MatFileWriter writer = new MatFileWriter();
            writer.write(file, list);
        } catch (IOException ex) {
            LOGGER.debug(null, ex);
        }
    }

    /**
     * Write data to file. Will Replace File if one exists. May need to build a
     * folder target/science/analysis.
     *
     * @param fileName
     * @param list
     */
    public static void storeToTestAnalysis(String fileName, List<MLArray> list) {

        Path path = Paths.get("target", "science", "analysis", fileName);

        try {

            if (Files.exists(path)) {
                Files.delete(path);
                Files.createFile(path);
            } else {
                Files.createFile(path);
            }

            MatFileWriter writer = new MatFileWriter();
            writer.write(path.toFile(), list);

            LOGGER.info("File Written To :", path.toFile().getName());
        } catch (IOException ex) {

            LOGGER.error("Unable to store to file:", ex);

        }

    }

    /**
     * Write data to file.Will Replace File if one exists. May need to build a
     * folder target/science/test.
     *
     * @param fileName
     * @param list
     */
    public static void storeToTest(String fileName, List<MLArray> list) {

        Path path = Paths.get("target", "science", "test", fileName);

        try {

            Files.createDirectories(path);
            Files.deleteIfExists(path);
            Files.createFile(path);
            MatFileWriter writer = new MatFileWriter();
            writer.write(path.toFile(), list);

            LOGGER.info("File Written To :", path.toFile().getName());
        } catch (IOException ex) {

            LOGGER.error("Unable to store to file:", ex);

        }

    }
    
    
        /**
     * Write data to file.Will Replace File if one exists. May need to build a
     * folder target/science/test.
     *
     * @param fileName
     * @param list
     */
    public static void storeToFinal(String fileName, List<MLArray> list) {

        Path path = Paths.get("science", fileName);

        try {

            Files.createDirectories(path);
            Files.deleteIfExists(path);
            Files.createFile(path);
            MatFileWriter writer = new MatFileWriter();
            writer.write(path.toFile(), list);

            LOGGER.info("File Written To :", path.toFile().getName());
        } catch (IOException ex) {

            LOGGER.error("Unable to store to file:", ex);

        }

    }
}
