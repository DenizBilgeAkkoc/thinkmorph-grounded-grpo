"""Trial-only HTTP/2 transport. Unmodified production Gemini request/retry logic.

Credentials enter curl through an inherited anonymous pipe, never argv/disk.
Request bodies are passed unchanged on stdin. Curl verifies Google TLS normally.
"""
import io,os,subprocess,tempfile,urllib.request,urllib.error,urllib.parse
from email.parser import Parser

class Response(io.BytesIO):
    def __init__(self,body,status,headers):
        super().__init__(body);self.status=status;self.headers=headers

def install():
    original=urllib.request.urlopen
    def urlopen(request,timeout=None,**kwargs):
        url=request.full_url if isinstance(request,urllib.request.Request) else request
        if urllib.parse.urlsplit(url).hostname!='generativelanguage.googleapis.com':
            return original(request,timeout=timeout,**kwargs)
        if not isinstance(request,urllib.request.Request):request=urllib.request.Request(url)
        assert not kwargs,kwargs.keys()
        def quote(value):
            assert not any(c in value for c in '\r\n\x00')
            return '"'+value.replace('\\','\\\\').replace('"','\\"')+'"'
        config=''.join('header = '+quote(k+': '+v)+'\n' for k,v in request.header_items()).encode()
        assert len(config)<4096
        readfd,writefd=os.pipe()
        try:
            os.write(writefd,config);os.close(writefd);writefd=None
            with tempfile.TemporaryFile() as headers:
                command=['/usr/bin/curl','--http2','--max-time',str(timeout or 180),'--silent','--show-error',
                         '--config',f'/proc/self/fd/{readfd}', '--dump-header',f'/proc/self/fd/{headers.fileno()}',
                         '--request',request.get_method(),'--write-out','\n%{http_code}',url]
                if request.data is not None:command+=['--data-binary','@-']
                curl_env=dict(os.environ);curl_env.pop('LD_LIBRARY_PATH',None)
                try:result=subprocess.run(command,input=request.data or b'',capture_output=True,env=curl_env,
                        pass_fds=(readfd,headers.fileno()),timeout=(timeout or 180)+10)
                except subprocess.TimeoutExpired:raise urllib.error.URLError('HTTP/2 transport timed out') from None
                headers.seek(0);header_text=headers.read().decode('iso-8859-1')
            if result.returncode:
                # Do not expose curl's error text, which may include user inputs.
                raise urllib.error.URLError('HTTP/2 transport failed, curl exit '+str(result.returncode))
            body,status=result.stdout.rsplit(b'\n',1);status=int(status)
            blocks=header_text.replace('\r\n','\n').strip().split('\n\n')
            parsed=Parser().parsestr('\n'.join(blocks[-1].splitlines()[1:])+'\n')
            if not 200<=status<300:raise urllib.error.HTTPError(url,status,'HTTP/2 response',parsed,io.BytesIO(body))
            return Response(body,status,parsed)
        finally:
            os.close(readfd)
            if writefd is not None:os.close(writefd)
    urllib.request.urlopen=urlopen
