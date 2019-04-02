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
package fit.astro.vsa.common.utilities.math.handling.sigfig;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SignificantDigitsTest {
    
    public SignificantDigitsTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

   
     @Test
     public void testSignificantDigits() {
     
         double test = 0.0010223;
         double testSigDig = SignificantDigits.
                 roundSignificantDigitsError(test, 3);
         
         assertEquals(0.00102, testSigDig, 1e-10);
         
         double testError = SignificantDigits.
                 roundToPrecision(test, 0.00002);
     
         assertEquals(0.001022, testError, 1e-10);
     }
     
     @Test
     public void testSignificantDigitsErrorFunction() {
     
         double test = 0.0010223;
         
         SignificantDigitsErrorFunction digitsErrorFunction 
                 = new SignificantDigitsErrorFunction(2);
                 
         
         double testSigDig = digitsErrorFunction.value(test);
         
         assertEquals(0.0010, testSigDig, 1e-10);
     
     }
     
  
}
