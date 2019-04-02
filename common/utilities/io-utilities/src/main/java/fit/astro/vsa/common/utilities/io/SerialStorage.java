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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Convenience method for using ObjectOuputStream to save an object to file.
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SerialStorage {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(SerialStorage.class);

    //Empty Constructor
    private SerialStorage(){
        
    }
    
    /**
     * Serialize object and store to file.
     *
     * @param serialObject
     * @param path
     * @return
     * @throws java.io.FileNotFoundException
     */
    public static boolean storeSerialObject(Object serialObject,
            File path) throws FileNotFoundException {

        // Remove old Object
        if (path.exists()) {
            path.delete();
        }

        // Make a new file
        try {
            path.createNewFile();
        } catch (IOException ex) {
            LOGGER.error(null, ex);
        }

        // Get Stream
        FileOutputStream fout = new FileOutputStream(path);
        boolean isSuccessful = Boolean.FALSE;
        try (ObjectOutputStream oos = new ObjectOutputStream(fout)) {
            // Write a object
            oos.writeObject(serialObject);
            oos.flush();
            oos.close();

            LOGGER.info("Stored Object to File: " + path.getAbsolutePath());
            isSuccessful = Boolean.TRUE;
        } catch (IOException ex) {
            LOGGER.error(null, ex);
        }
        return isSuccessful;
    }

}
