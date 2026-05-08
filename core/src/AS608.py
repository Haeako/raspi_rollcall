#!/usr/bin/env python
from dataclasses import dataclass
import time
from pyfingerprint.pyfingerprint import PyFingerprint

def Error_handler():
    return

class AS608_HAL:
    def __init__(self):
        try:
            self.sensor = PyFingerprint('/dev/ttyAMA0', 57600, 0xFFFFFFFF, 0x00000000)

            if ( self.sensor.verifyPassword() == False ):
                return self.ReturnValue(success=False, message='The given fingerprint sensor password is wrong!')
            print('Storage: ' + str(self.sensor.getTemplateCount()) + '/'+ str(self.sensor.getStorageCapacity()))

        except Exception as e:
            return self.ReturnValue(success=False, message='The fingerprint sensor could not be initialized! Exception message: ' + str(e))
            self.sensor = None
            Error_handler()
        return
    
    @dataclass
    class ReturnValue:
        success: bool
        message: str

    def enroll(self):
        try:
            ## Wait that finger is read
            while ( self.sensor.readImage() == False ):
                pass

            ## Converts read image to characteristics and stores it in charbuffer 1
            self.sensor.convertImage(0x01)

            ## Checks if finger is already enrolled
            result = self.sensor.searchTemplate()
            positionNumber = result[0]

            if ( positionNumber >= 0 ):
                return self.ReturnValue(success=False, message='Template already exists')

            time.sleep(2)
            while ( self.sensor.readImage() == False ):
                pass

            self.sensor.convertImage(0x02)

            if ( self.sensor.compareCharacteristics() == 0 ):
                return self.ReturnValue(success=False, message='Fingers do not match')

            self.sensor.createTemplate()

            ## Saves template at new position number
            positionNumber = self.sensor.storeTemplate()
            return self.ReturnValue(success=True, message='Finger enrolled successfully at position #' + str(positionNumber))

        except Exception as e:
            return self.ReturnValue(success=False, message='Operation failed! Exception message: ' + str(e))
    def search(self):
        try:
            ## Wait that finger is read
            while ( self.sensor.readImage() == False ):
                pass

            ## Converts read image to characteristics and stores it in charbuffer 1
            self.sensor.convertImage(0x01)

            ## Searchs template
            result = self.sensor.searchTemplate()
            positionNumber = result[0]
            accuracyScore = result[1]

            if ( positionNumber == -1 ):
                return self.ReturnValue(success=False, message='No match found!')
            else:
                return self.ReturnValue(success=True, message='Found template at position #' + str(positionNumber) + ' with accuracy score of ' + str(accuracyScore))

        except Exception as e:
            return self.ReturnValue(success=False, message='Operation failed! Exception message: ' + str(e))
    def delete(self, positionNumber):
        try:
            if ( self.sensor.deleteTemplate(positionNumber) == True ):
                return self.ReturnValue(success=True, message='Template deleted successfully!')
            else:
                return self.ReturnValue(success=False, message='Failed to delete template')

        except Exception as e:
            return self.ReturnValue(success=False, message='Operation failed! Exception message: ' + str(e))
    def empty(self):
        try:
            if ( self.sensor.clearDatabase() == True ):
                return self.ReturnValue(success=True, message='Database cleared successfully!')
            else:
                return self.ReturnValue(success=False, message='Failed to clear database')

        except Exception as e:
            return self.ReturnValue(success=False, message='Operation failed! Exception message: ' + str(e))

def main():
    hal = AS608_HAL()
    result = hal.enroll()
    print(result.message)
    time.sleep(2)
    result = hal.search()
    print(result.message)
    hal.delete(0)
if __name__ == "__main__":    
    main()